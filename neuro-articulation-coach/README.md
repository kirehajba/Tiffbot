# 🧠 NeuroSpeak — Neuroplasticity & Articulation Coach

A daily communication trainer that uses AI coaching and neuroplasticity principles to rewire how you speak. Five drills, five minutes a day: warm up your articulators, think on your feet, cut the fluff, automate structure, and grow your vocabulary — with instant scored feedback on every rep and progress tracking over time.

## The science

Each drill is built on an established learning mechanism:

| Drill | Skill | Mechanism |
|-------|-------|-----------|
| 🔥 Warm-Up | Diction & pace | Motor cortex priming — repeated precise articulation myelinates speech-motor circuits |
| 🎤 Impromptu | Fluency under pressure | Retrieval practice — generating answers in real time is the strongest driver of durable learning |
| ✂️ Precision | Concision | Error-driven learning — comparing your rewrite to a tighter version creates the prediction errors the brain learns from |
| 🏗️ Structure | Organized thinking | Chunking — frameworks (PREP, STAR, What/So What/Now What, Rule of Three) free working memory for content |
| 📚 Vocabulary | Lexical range | Elaborative encoding — words you *produce* become words you can reach for mid-sentence |

Consistency beats intensity: neuroplasticity rewards daily frequency, and sleep consolidates each rep. The app tracks your streak, scores every drill on **clarity, concision, structure, and impact**, and tells you which skill to focus on next.

## Features

- **🎭 AI roleplay** — spar with a Skeptical CFO, Tough Interviewer, Resistant Team, or Distracted Executive; they push back turn by turn, then you get coached on how you handled the pressure
- **🎯 My Meeting** — describe a real upcoming situation and get a personalized training circuit: the 3 hard questions that audience will ask, the best-fit framework, an opener tip, and per-answer coaching
- **Voice recording** — record your answers, get verbatim transcription, filler-word counts, and words-per-minute pace analysis
- **⏱ Pressure mode** — 60-second countdown on impromptu drills and a live elapsed timer while recording
- **AI coaching** — every rep is scored 1–10 across four skills, with strengths, highest-leverage fixes, and an upgraded version of what you said
- **Tongue-twister diction scoring** — word-level accuracy against the target
- **Daily rotation** — exercises rotate deterministically each day (spaced variation), with a shuffle button when you want more reps
- **Progress dashboard** — streak, drill history, per-skill averages, score-over-time chart, week-over-week trend, and a recommended focus area

## Two ways to run it

1. **GitHub Pages (zero install)** — a fully static, browser-only version lives in [`docs/index.html`](docs/index.html). Enable Pages (Settings → Pages → Deploy from branch → `main` / `docs`) and open the site. Paste your OpenAI API key in the ⚙️ settings — it's stored only in your browser's localStorage and sent only to api.openai.com. Progress is tracked in localStorage.
2. **Streamlit app (local)** — the full Python version below.

## Quick start (Streamlit version)

```bash
git clone https://github.com/kirehajba/neuro-articulation-coach.git
cd neuro-articulation-coach
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your OpenAI API key
streamlit run app.py
```

Open http://localhost:8501 and start with the Warm-Up tab.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`
3. Add `OPENAI_API_KEY` under **App settings → Secrets**

Note: progress is stored in a local `.progress/` folder, which is ephemeral on Streamlit Cloud — for durable long-term tracking, run locally.

## Project structure

```
neuro-articulation-coach/
├── app.py            # Streamlit entry point — layout, styling, sidebar
├── trainer.py        # Drills, AI feedback, audio analysis, progress tracking
├── requirements.txt
├── .env.example
└── README.md
```

## How feedback works

Your response (spoken → transcribed verbatim, or typed) is sent to GPT-4o with a coaching system prompt. The model returns structured JSON: per-skill scores, what worked, the highest-leverage improvements, and a rewritten stronger version. Spoken answers additionally get local analysis: pace (wpm), filler-word count, and — for tongue twisters — word-level diction accuracy.
