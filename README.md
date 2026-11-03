# 📅 AI Study Planner

Enter your subjects, exam dates, weak topics and available hours. The app builds an editable study timetable.

## Features
- Add subjects with exam dates, a confidence level (1-5) and weak topics.
- Set the hours you can study on each weekday, the session length and breaks.
- The timetable gives more time to weak subjects and exams that are closer, and covers your weak topics first.
- The last two days before each exam are used for a past paper and final revision.
- **Fully editable**: change any cell, add or delete rows, tick sessions as Done.
- Day-by-day view, hours per subject chart, completion progress, exam countdown.
- Download as CSV (Excel) or a calendar `.ics` file for Google Calendar, Outlook or Apple Calendar.
- Optional AI (Groq): suggests topics for each subject and gives personal study tips.
- Nothing is stored permanently; everything lives in your browser session.

## How the timetable is made
The schedule is made by code in `app.py`, not by AI, so it is predictable and works without a key.
Each study slot goes to the subject that is furthest behind its share, where the share depends on how weak you are
and how close the exam is. The AI (if enabled) only suggests topic names and tips.

## Files
| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app |
| `requirements.txt` | Python packages (streamlit, pandas, groq) |
| `.gitignore` | Keeps secrets out of GitHub |

## Groq API key (optional)
1. Create a free key at https://console.groq.com (API Keys → Create API Key).
2. On Streamlit Community Cloud: app **Settings → Secrets**, paste one line with straight quotes:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. Save and **Reboot app**.

Locally, put the same line in `.streamlit/secrets.toml` (already in `.gitignore`). Never put the key in `app.py`.

## Run on your computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```

## Tips
- Replace the three example subjects with your own before clicking **Create my timetable**.
- Dates use the server's idea of "today". If your day looks off by one, change **Start studying from**.
- Editing the subjects table later does not change an existing timetable; click **Create my timetable** again.
  (Creating a new timetable replaces your manual edits.)

## Storage note
To keep timetables between visits you would need a database. Use a free hosted one such as Neon PostgreSQL;
Streamlit Community Cloud erases local files (including SQLite) when the app restarts. Meanwhile, download the CSV.
