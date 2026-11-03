"""AI Study Planner - a Streamlit app.

Enter your subjects, exam dates, weak topics and available hours. The app builds an
editable study timetable. The schedule itself is made by built-in code, so it works
without AI. The Groq API is optional and adds topic suggestions and study tips.
"""

import json
import os
import re
from datetime import date, datetime, time, timedelta
from itertools import cycle

import pandas as pd
import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None

# Safety guard: if this file was accidentally pasted twice, ignore the second copy.
if globals().get("_PLANNER_LOADED"):
    st.stop()
_PLANNER_LOADED = True

st.set_page_config(page_title="AI Study Planner", page_icon="📅", layout="wide")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
PREFERRED_MODELS = [
    GROQ_MODEL,
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]
NON_CHAT_WORDS = ("whisper", "guard", "tts", "orpheus", "embed", "playai", "safeguard", "compound")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DEFAULT_HOURS = [2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 3.0]
ACTIVITIES = ["Learn", "Practice", "Revise", "Past paper", "Rest", "Other"]
DEFAULT_TOPICS = ["Core concepts", "Key definitions and formulas", "Practice questions",
                  "Summary notes", "Review past mistakes"]
MAX_SUBJECTS = 12
MAX_DAYS = 180


# ----------------------------------------------------------------------------
# Groq helpers (optional AI)
# ----------------------------------------------------------------------------
def get_api_key():
    """Read the Groq key from Secrets or the environment, cleaning common paste mistakes."""
    key = None
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    key = str(key).strip().strip("\"'\u201c\u201d\u2018\u2019").strip()
    key = re.sub(r"\s+", "", key)
    return key or None


def pick_model(client):
    try:
        ids = {m.id for m in client.models.list().data}
    except Exception:
        return None
    for name in PREFERRED_MODELS:
        if name in ids:
            return name
    for name in sorted(ids):
        if not any(word in name.lower() for word in NON_CHAT_WORDS):
            return name
    return None


def model_is_missing(exc):
    text = str(exc).lower()
    return "NotFound" in type(exc).__name__ or "decommission" in text or "does not exist" in text or (
        "model" in text and "not found" in text
    )


def ask_ai(messages, json_mode=False, max_tokens=900):
    """Returns (text, error_message). Switches model automatically if one is retired."""
    key = get_api_key()
    if Groq is None or not key:
        return None, "The AI is not set up. Add GROQ_API_KEY in the app Secrets."
    try:
        client = Groq(api_key=key, timeout=40.0, max_retries=1)
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}

        def run(model):
            tokens = max_tokens * 3 if "gpt-oss" in model or "qwen3" in model else max_tokens
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.3, max_tokens=tokens, **extra
            )
            return (response.choices[0].message.content or "").strip()

        model = st.session_state.get("groq_model", GROQ_MODEL)
        try:
            return run(model), None
        except Exception as exc:
            if not model_is_missing(exc):
                raise
            replacement = pick_model(client)
            if not replacement or replacement == model:
                return None, "No working Groq chat model was found for your key."
            st.session_state["groq_model"] = replacement
            return run(replacement), None
    except Exception as exc:
        name = type(exc).__name__
        if "RateLimit" in name:
            return None, "The AI service is busy (rate limit). Wait a minute and try again."
        if "Timeout" in name or "Connection" in name:
            return None, "The AI service did not respond in time. Please try again."
        if "Authentication" in name or "PermissionDenied" in name:
            return None, ("The Groq API key was rejected. Create a new key at console.groq.com, paste it as one "
                          "line: GROQ_API_KEY = \"gsk_...\" (straight quotes), save, then reboot the app.")
        return None, "Something went wrong contacting the AI service. Please try again."


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def ai_topics(subjects):
    """Ask the AI for a topic list per subject. Returns (dict name -> topics, error)."""
    lines = [f"- {s['name']} (weak topics: {', '.join(s['weak']) or 'none given'})" for s in subjects]
    messages = [
        {"role": "system", "content": (
            "You help students plan revision. Return ONLY valid JSON of the form "
            '{"subjects": {"Subject name": ["topic 1", "topic 2"]}}. For each subject give 6 to 8 specific, '
            "commonly examined study topics (short phrases, under 8 words each). Include the student's weak "
            "topics if given. Use the exact subject names provided.")},
        {"role": "user", "content": "Subjects:\n" + "\n".join(lines)},
    ]
    text, error = ask_ai(messages, json_mode=True)
    if error:
        return {}, error
    data = parse_json(text)
    raw = data.get("subjects") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}, "The AI reply could not be read, so default topics were used."
    out = {}
    for name, topics in raw.items():
        if isinstance(topics, list):
            clean = [str(t).strip()[:70] for t in topics if str(t).strip()]
            if clean:
                out[str(name)] = clean[:10]
    return out, None


# ----------------------------------------------------------------------------
# Reading the inputs
# ----------------------------------------------------------------------------
def split_topics(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    parts = re.split(r"[,;\n]", str(text))
    return [p.strip()[:60] for p in parts if p.strip()][:10]


def clean_subjects(df, start):
    """Turn the subjects table into a list of dicts. Returns (subjects, messages)."""
    subjects, messages, seen = [], [], set()
    for _, row in df.iterrows():
        name = str(row.get("Subject") or "").strip()
        if not name or name.lower() == "nan":
            continue
        if name.lower() in seen:
            messages.append(("warning", f"'{name}' is listed twice. Only the first one is used."))
            continue
        exam = pd.to_datetime(row.get("Exam date"), errors="coerce")
        if pd.isna(exam):
            messages.append(("error", f"'{name}' needs an exam date."))
            continue
        exam = exam.date()
        if exam <= start:
            messages.append(("warning", f"'{name}' was skipped: its exam date must be after your start date."))
            continue
        conf = pd.to_numeric(row.get("Confidence"), errors="coerce")
        conf = 3 if pd.isna(conf) else int(min(5, max(1, round(float(conf)))))
        seen.add(name.lower())
        subjects.append({"name": name[:60], "exam": exam, "conf": conf, "weak": split_topics(row.get("Weak topics"))})
    if len(subjects) > MAX_SUBJECTS:
        messages.append(("warning", f"Only the first {MAX_SUBJECTS} subjects are used."))
        subjects = subjects[:MAX_SUBJECTS]
    return subjects, messages


# ----------------------------------------------------------------------------
# The scheduler
# ----------------------------------------------------------------------------
def topic_order(subject, suggested):
    weak = subject["weak"]
    weak_lower = {w.lower() for w in weak}
    base = [t for t in (suggested or DEFAULT_TOPICS) if t.lower() not in weak_lower]
    return (weak + base + weak) or DEFAULT_TOPICS  # weak topics come first and again later


def build_plan(subjects, start, hours, session_min, break_min, start_time, suggested):
    """Return a list of study session rows. Weak subjects and close exams get more time."""
    horizon = min(max(s["exam"] for s in subjects), start + timedelta(days=MAX_DAYS))
    queues = {s["name"]: cycle(topic_order(s, suggested.get(s["name"]))) for s in subjects}
    assigned = {s["name"]: 0 for s in subjects}
    rows = []

    day = start
    while day < horizon:
        slots = int((hours[day.weekday()] * 60) // session_min)
        active = [s for s in subjects if day < s["exam"]]
        if slots and active:
            picks, previous = [], None
            for _ in range(slots):
                scored = []
                for s in active:
                    days_left = (s["exam"] - day).days
                    weight = ((6 - s["conf"]) + 0.5) * (1 + 6 / days_left)
                    scored.append(((assigned[s["name"]] + 0.5) / weight, s))
                scored.sort(key=lambda item: item[0])
                choice = scored[0]
                if previous and choice[1]["name"] == previous:  # avoid the same subject twice in a row
                    for candidate in scored[1:]:
                        if candidate[0] <= choice[0] * 1.5:
                            choice = candidate
                            break
                picks.append(choice[1])
                assigned[choice[1]["name"]] += 1
                previous = choice[1]["name"]

            total = slots * session_min + (slots - 1) * break_min
            begin = datetime.combine(day, start_time)
            latest = datetime.combine(day, time(23, 59)) - timedelta(minutes=total)
            if begin > latest:
                begin = max(latest, datetime.combine(day, time(6, 0)))

            for s in picks:
                days_left = (s["exam"] - day).days
                progress = (day - start).days / max(1, (s["exam"] - start).days)
                if days_left == 1:
                    topic, activity = "Final revision: key points and weak topics", "Revise"
                elif days_left == 2:
                    topic, activity = "Timed past paper or practice test", "Past paper"
                else:
                    topic = next(queues[s["name"]])
                    activity = "Learn" if progress < 0.5 else ("Practice" if progress < 0.8 else "Revise")
                rows.append({"Date": day, "Start": begin.time().replace(second=0, microsecond=0),
                             "Minutes": session_min, "Subject": s["name"], "Topic": topic,
                             "Activity": activity, "Done": False, "Notes": ""})
                begin += timedelta(minutes=session_min + break_min)
        day += timedelta(days=1)
    return rows


# ----------------------------------------------------------------------------
# Working with the edited timetable
# ----------------------------------------------------------------------------
def to_time(value):
    if isinstance(value, time):
        return value
    try:
        return pd.to_datetime(str(value)).time().replace(second=0, microsecond=0)
    except Exception:
        return time(9, 0)


def clean_plan(df, default_minutes):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"])
    d["Date"] = d["Date"].dt.date
    d["Start"] = d["Start"].apply(to_time)
    d["Minutes"] = pd.to_numeric(d["Minutes"], errors="coerce").fillna(default_minutes).clip(5, 480).astype(int)
    d["Subject"] = d["Subject"].fillna("Other").astype(str)
    d["Topic"] = d["Topic"].fillna("").astype(str)
    d["Activity"] = d["Activity"].fillna("Learn").astype(str)
    d["Notes"] = d["Notes"].fillna("").astype(str)
    d["Done"] = d["Done"].fillna(False).astype(bool)
    return d.sort_values(["Date", "Start"]).reset_index(drop=True)


def end_time(day, start, minutes):
    return (datetime.combine(day, start) + timedelta(minutes=int(minutes))).time()


def ics_escape(text):
    return str(text).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def build_ics(plan, exams):
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AI Study Planner//EN", "CALSCALE:GREGORIAN"]
    for i, row in plan.iterrows():
        begin = datetime.combine(row["Date"], row["Start"])
        finish = begin + timedelta(minutes=int(row["Minutes"]))
        title = f"{row['Subject']}: {row['Topic']}" if row["Topic"] else row["Subject"]
        lines += ["BEGIN:VEVENT", f"UID:study-{i}-{begin:%Y%m%d%H%M}@aistudyplanner", f"DTSTAMP:{stamp}",
                  f"DTSTART:{begin:%Y%m%dT%H%M%S}", f"DTEND:{finish:%Y%m%dT%H%M%S}",
                  f"SUMMARY:{ics_escape(title)}", f"DESCRIPTION:{ics_escape(row['Activity'])}", "END:VEVENT"]
    for i, (name, exam_day) in enumerate(exams):
        lines += ["BEGIN:VEVENT", f"UID:exam-{i}-{exam_day:%Y%m%d}@aistudyplanner", f"DTSTAMP:{stamp}",
                  f"DTSTART;VALUE=DATE:{exam_day:%Y%m%d}", f"DTEND;VALUE=DATE:{exam_day + timedelta(days=1):%Y%m%d}",
                  f"SUMMARY:{ics_escape('EXAM: ' + name)}", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("📅 AI Study Planner")
st.write("Enter your subjects, exam dates, weak topics and free hours. Get a timetable you can edit.")
st.caption("🔒 Privacy: your details are used only in this browser session and are not stored permanently.")

today = date.today()
if "subjects_base" not in st.session_state:
    st.session_state["subjects_base"] = pd.DataFrame({
        "Subject": ["Mathematics", "Physics", "English"],
        "Exam date": [today + timedelta(days=21), today + timedelta(days=28), today + timedelta(days=35)],
        "Confidence": [2, 3, 4],
        "Weak topics": ["Integration, Probability", "Electricity", ""],
    })

ai_ready = Groq is not None and bool(get_api_key())

st.subheader("1. Your subjects")
st.caption("Add one row per subject. Confidence: 1 = I struggle, 5 = I am strong. Separate weak topics with commas. "
           "Replace the example rows with your own.")
subjects_df = st.data_editor(
    st.session_state["subjects_base"], key="subjects_editor", num_rows="dynamic", hide_index=True,
    column_config={
        "Subject": st.column_config.TextColumn("Subject", required=True, max_chars=60),
        "Exam date": st.column_config.DateColumn("Exam date", format="YYYY-MM-DD", required=True),
        "Confidence": st.column_config.NumberColumn("Confidence (1-5)", min_value=1, max_value=5, step=1, default=3),
        "Weak topics": st.column_config.TextColumn("Weak topics (comma separated)", max_chars=300),
    },
)

st.subheader("2. Your available hours")
cols = st.columns(7)
hours = []
for i, col in enumerate(cols):
    hours.append(col.number_input(DAY_NAMES[i][:3], min_value=0.0, max_value=12.0, value=DEFAULT_HOURS[i],
                                  step=0.5, key=f"hours_{i}"))
st.caption("Hours you can study on each weekday. Use 0 for a day off.")

st.subheader("3. Options")
c1, c2, c3, c4 = st.columns(4)
start = c1.date_input("Start studying from", value=today, min_value=today - timedelta(days=30), key="start_date")
session_min = c2.selectbox("Session length (minutes)", [30, 45, 60, 90], index=2)
break_min = c3.selectbox("Break between sessions (minutes)", [0, 5, 10, 15], index=2)
start_time = c4.time_input("Daily start time", value=time(16, 0), step=900)
use_ai = st.checkbox("✨ Use AI to suggest topics for each subject", value=ai_ready, disabled=not ai_ready,
                     help="Needs a Groq key. Without it the planner uses your weak topics plus general study topics.")
if not ai_ready:
    st.caption("AI is off because no Groq key was found. The timetable still works; see the README to enable AI.")

if st.button("📅 Create my timetable", type="primary"):
    subjects, messages = clean_subjects(subjects_df, start)
    for level, text in messages:
        getattr(st, level)(text)
    if not subjects:
        st.error("Please enter at least one subject with an exam date after your start date.")
    elif any(level == "error" for level, _ in messages):
        st.error("Please fix the problems above and try again.")
    elif sum(hours) == 0 or all(int((h * 60) // session_min) == 0 for h in hours):
        st.error("You have no study time. Enter some available hours (at least one session long).")
    else:
        suggested = {}
        if use_ai and ai_ready:
            with st.spinner("Asking the AI for topics..."):
                suggested, error = ai_topics(subjects)
            if error:
                st.warning(error)
        rows = build_plan(subjects, start, hours, session_min, break_min, start_time, suggested)
        if not rows:
            st.error("No study sessions could be created. Check your dates and available hours.")
        else:
            counts = pd.Series([r["Subject"] for r in rows]).value_counts()
            for s in subjects:
                if counts.get(s["name"], 0) < 3:
                    st.warning(f"'{s['name']}' only got {counts.get(s['name'], 0)} session(s). "
                               "Add more hours or a longer time before the exam.")
            st.session_state.update(
                plan_base=pd.DataFrame(rows), plan_version=st.session_state.get("plan_version", 0) + 1,
                exams=[(s["name"], s["exam"]) for s in subjects], subject_info=subjects,
                default_minutes=session_min, tips=None,
            )
            st.success(f"Timetable created: {len(rows)} sessions. Scroll down to edit it.")

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if "plan_base" not in st.session_state:
    st.info("Fill in the three steps above and click **Create my timetable**.")
    st.stop()

version = st.session_state["plan_version"]
exams = st.session_state["exams"]
names = [name for name, _ in exams]
st.divider()
st.header("Your timetable")

tab_edit, tab_day, tab_summary, tab_download = st.tabs(["✏️ Edit timetable", "🗓️ Day by day", "📊 Summary", "⬇️ Download"])

with tab_edit:
    st.caption("Click any cell to change it. Add rows at the bottom, delete rows by selecting them, "
               "and tick **Done** as you study. Your edits are kept until you create a new timetable.")
    edited = st.data_editor(
        st.session_state["plan_base"], key=f"plan_editor_{version}", num_rows="dynamic", hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="ddd D MMM YYYY", required=True),
            "Start": st.column_config.TimeColumn("Start", format="HH:mm", step=300),
            "Minutes": st.column_config.NumberColumn("Minutes", min_value=5, max_value=480, step=5,
                                                     default=st.session_state["default_minutes"]),
            "Subject": st.column_config.SelectboxColumn("Subject", options=names + ["Other"], required=True),
            "Topic": st.column_config.TextColumn("Topic", width="large"),
            "Activity": st.column_config.SelectboxColumn("Activity", options=ACTIVITIES),
            "Done": st.column_config.CheckboxColumn("Done", default=False),
            "Notes": st.column_config.TextColumn("Notes"),
        },
    )

plan = clean_plan(edited, st.session_state["default_minutes"])

with tab_day:
    if plan.empty:
        st.info("Your timetable is empty. Add rows in the Edit tab.")
    else:
        show_past = st.checkbox("Show past days", value=False)
        view = plan if show_past else plan[plan["Date"] >= today]
        if view.empty:
            st.info("All sessions are in the past. Tick 'Show past days' to see them.")
        limit = st.slider("Days to show", 3, 60, 14) if not view.empty else 0
        for day, group in list(view.groupby("Date"))[:limit]:
            exam_here = [n for n, d in exams if d == day]
            st.markdown(f"**{day:%A, %d %B %Y}**" + (f"  🎯 EXAM: {', '.join(exam_here)}" if exam_here else ""))
            for _, r in group.iterrows():
                mark = "✅" if r["Done"] else "⬜"
                topic = f": {r['Topic']}" if r["Topic"] else ""
                st.markdown(f"{mark} {r['Start']:%H:%M}-{end_time(r['Date'], r['Start'], r['Minutes']):%H:%M} "
                            f"**{r['Subject']}**{topic} _({r['Activity']})_")

with tab_summary:
    if plan.empty:
        st.info("Your timetable is empty.")
    else:
        total_min, done_min = int(plan["Minutes"].sum()), int(plan.loc[plan["Done"], "Minutes"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Sessions", len(plan))
        m2.metric("Total study time", f"{total_min / 60:.1f} h")
        m3.metric("Completed", f"{(100 * done_min / total_min) if total_min else 0:.0f}%")
        st.progress(done_min / total_min if total_min else 0.0)

        rows = []
        for name, exam_day in exams:
            mine = plan[plan["Subject"] == name]
            rows.append({"Subject": name, "Exam date": exam_day, "Days left": (exam_day - today).days,
                         "Planned hours": round(mine["Minutes"].sum() / 60, 1),
                         "Done hours": round(mine.loc[mine["Done"], "Minutes"].sum() / 60, 1)})
        summary = pd.DataFrame(rows)
        st.dataframe(summary, hide_index=True)
        st.bar_chart(summary.set_index("Subject")["Planned hours"])

        st.subheader("AI study tips")
        if not ai_ready:
            st.caption("Add a Groq key to get personalised tips (see the README).")
        elif st.button("✨ Get tips for my plan"):
            info = "\n".join(
                f"- {s['name']}: exam {s['exam']}, confidence {s['conf']}/5, weak topics: {', '.join(s['weak']) or 'none'}, "
                f"planned hours: {summary.loc[summary['Subject'] == s['name'], 'Planned hours'].sum()}"
                for s in st.session_state["subject_info"])
            with st.spinner("Thinking..."):
                text, error = ask_ai([
                    {"role": "system", "content": "You are a supportive study coach. Give practical, specific advice."},
                    {"role": "user", "content": "Here is my study plan:\n" + info +
                     "\n\nGive 5 short, practical tips tailored to these subjects and my weak topics, "
                     "plus one tip for exam week. Use a numbered list. Under 220 words."}])
            if error:
                st.error(error)
            else:
                st.session_state["tips"] = text
        if st.session_state.get("tips"):
            st.markdown(st.session_state["tips"])

with tab_download:
    if plan.empty:
        st.info("Your timetable is empty.")
    else:
        export = plan.copy()
        export["End"] = [end_time(r["Date"], r["Start"], r["Minutes"]) for _, r in export.iterrows()]
        export = export[["Date", "Start", "End", "Minutes", "Subject", "Topic", "Activity", "Done", "Notes"]]
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Timetable (CSV, opens in Excel)", export.to_csv(index=False).encode("utf-8-sig"),
                           "study_timetable.csv", "text/csv")
        d2.download_button("⬇️ Calendar file (.ics)", build_ics(plan, exams), "study_timetable.ics", "text/calendar")
        st.caption("The .ics file can be imported into Google Calendar, Outlook or Apple Calendar. "
                   "It includes your study sessions and exam days.")
