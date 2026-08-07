"""
Neuroplasticity & Articulation Coach — daily communication drills with AI
feedback and progress tracking.

Each drill targets a distinct communication skill and is built on established
neuroplasticity principles: deliberate practice, retrieval practice, spaced
repetition, and immediate feedback. Rendered by app.py.
"""

from __future__ import annotations

import datetime
import difflib
import io
import json
import os
import random
import wave

import streamlit as st

PROGRESS_DIR = os.path.join(os.path.dirname(__file__), ".progress")
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "articulation_progress.json")

FEEDBACK_MODEL = "gpt-4o"
TRANSCRIBE_MODEL = "gpt-4o-transcribe"

SKILLS = ["clarity", "concision", "structure", "impact"]

# ---------------------------------------------------------------------------
# Exercise banks
# ---------------------------------------------------------------------------

TONGUE_TWISTERS = [
    ("Red leather, yellow leather, red leather, yellow leather.", "easy"),
    ("Unique New York, unique New York, you know you need unique New York.", "easy"),
    ("She sells seashells by the seashore, and the shells she sells are seashells.", "medium"),
    ("Six sleek swans swam swiftly southwards.", "medium"),
    ("A proper copper coffee pot, a proper copper coffee pot.", "medium"),
    ("Peter Piper picked a peck of pickled peppers; a peck of pickled peppers Peter Piper picked.", "hard"),
    ("The sixth sick sheikh's sixth sheep's sick.", "hard"),
    ("Strategic statistics stipulate stronger structural standards.", "hard"),
    ("Executives execute exceptional expectations expertly and expediently.", "hard"),
    ("Brisk brave brigadiers brandished broad bright blades.", "hard"),
]

IMPROMPTU_TOPICS = [
    "Convince a skeptical executive team to fund your project in 60 seconds.",
    "Explain what you do at work to a curious 10-year-old.",
    "You just walked into the elevator with the CEO. Introduce yourself and your biggest current win.",
    "A colleague publicly challenged your idea in a meeting. Respond with composure.",
    "Announce a difficult change to your team while keeping morale up.",
    "Pitch yourself for a promotion in under a minute — no notes.",
    "Describe a failure you learned from, framed as a leadership story.",
    "Explain why clear communication matters more as you get more senior.",
    "Give a 60-second toast at a colleague's farewell party.",
    "Summarize your week for a stakeholder who has 45 seconds and no context.",
    "Persuade your team to adopt a tool or habit they're resisting.",
    "You have one minute to open a town hall meeting. Set the tone.",
    "Defend an unpopular but correct decision you made.",
    "Explain a complex idea from your field using only everyday language.",
    "Tell a stranger why your work matters to the world.",
]

CONCISION_CHALLENGES = [
    "At this point in time, we are currently in the process of conducting a review of the various different options that are available to us, and it is our intention to make a final decision at some point in the near future once we have had the opportunity to fully evaluate each and every one of the alternatives.",
    "I just wanted to reach out and touch base with you in order to see if you might possibly have some availability at some point this week to maybe hop on a quick call so we could discuss and talk through some of the thoughts and ideas I've been having recently about the project.",
    "Due to the fact that there has been a significant amount of confusion with regard to the new policy, it has been decided by the management team that an additional meeting will be scheduled for the purpose of providing further clarification to all of the employees who may be affected.",
    "In my personal opinion, I think that it would probably be a good idea for us to consider the possibility of perhaps starting the planning process earlier than we did last year, because of the fact that last year we ended up running out of time at the end.",
    "The reason why the project was delayed is because of the fact that there were a number of unforeseen and unexpected challenges that arose during the course of the implementation phase, which required the team to spend additional extra time resolving them before being able to move forward.",
    "It goes without saying that each and every member of the team should make an effort to try to attend the upcoming training session, which will serve to provide everyone with the necessary skills that will be needed going forward in the future.",
]

FRAMEWORKS = {
    "PREP": {
        "structure": "**P**oint → **R**eason → **E**xample → **P**oint (restated)",
        "description": "The fastest way to sound structured under pressure. State your point, give the reason, prove it with an example, land the point again.",
        "prompts": [
            "Should meetings default to 25 minutes instead of 30?",
            "Is remote work better for deep-focus roles?",
            "Should leaders share their failures openly with their teams?",
            "Is it better to be respected or liked as a manager?",
            "Should companies ban after-hours email?",
        ],
    },
    "STAR": {
        "structure": "**S**ituation → **T**ask → **A**ction → **R**esult",
        "description": "The interview and storytelling workhorse. Anchor the listener in context, then show what YOU did and what changed because of it.",
        "prompts": [
            "Tell me about a time you influenced a decision without authority.",
            "Describe a moment you turned around an underperforming situation.",
            "Tell me about a time you had to deliver bad news.",
            "Describe a conflict with a colleague and how you resolved it.",
            "Tell me about the achievement you're most proud of.",
        ],
    },
    "What / So What / Now What": {
        "structure": "**What** happened → **So what** does it mean → **Now what** should we do",
        "description": "The executive-update pattern. Facts first, implication second, recommendation last — the order senior audiences expect.",
        "prompts": [
            "Update leadership on a project that slipped two weeks.",
            "Brief your team on a competitor's surprising product launch.",
            "Report a customer complaint trend to your boss.",
            "Summarize the results of an experiment that failed.",
            "Present a budget overrun to the steering committee.",
        ],
    },
    "Rule of Three": {
        "structure": "One message → **three** supporting points → close",
        "description": "Audiences remember things in threes. Force yourself to pick exactly three supports — no more, no less — and your message sharpens itself.",
        "prompts": [
            "Why should someone join your team? Three reasons.",
            "What makes a great meeting? Three ingredients.",
            "Why do good employees quit? Three causes.",
            "What builds trust fastest? Three behaviors.",
            "What separates senior leaders from managers? Three differences.",
        ],
    },
}

VOCAB_WORDS = [
    ("articulate", "expressing ideas fluently and coherently"),
    ("galvanize", "to shock or excite someone into taking action"),
    ("cogent", "clear, logical, and convincing"),
    ("distill", "to extract the essential meaning from something"),
    ("catalyze", "to cause or accelerate a change"),
    ("succinct", "briefly and clearly expressed"),
    ("underscore", "to emphasize or draw attention to"),
    ("delineate", "to describe or portray precisely"),
    ("pragmatic", "dealing with things sensibly and realistically"),
    ("salient", "most noticeable or important"),
    ("elucidate", "to make something clear; explain"),
    ("juxtapose", "to place side by side for contrast"),
    ("consensus", "general agreement among a group"),
    ("leverage", "to use something to maximum advantage"),
    ("paradigm", "a typical example or pattern of something"),
]

NEURO_NOTES = {
    "warmup": (
        "**Why this works — motor cortex priming.** Rapid, precise articulation drills "
        "activate the motor pathways that control your lips, tongue, and jaw. Repeating "
        "them daily strengthens the myelin insulation around those circuits, making crisp "
        "diction automatic — so under pressure, your mouth keeps up with your mind."
    ),
    "impromptu": (
        "**Why this works — retrieval under pressure.** Speaking with no script forces "
        "your brain to retrieve and organize ideas in real time. This is *retrieval "
        "practice* — the single most powerful driver of durable learning. Mild time "
        "pressure adds just enough stress to trigger norepinephrine release, which tags "
        "the practice as important and accelerates rewiring."
    ),
    "precision": (
        "**Why this works — error-driven learning.** Comparing your rewrite against a "
        "tighter version creates a prediction error — the gap between what you produced "
        "and what's possible. Dopamine-driven error signals are exactly what the brain "
        "uses to update its internal models. Over weeks, verbose habits get pruned and "
        "concise phrasing becomes your default."
    ),
    "structure": (
        "**Why this works — chunking.** Frameworks like PREP and STAR let your brain "
        "chunk a complex answer into a small number of familiar slots, freeing working "
        "memory for content instead of scaffolding. With spaced repetition, the "
        "framework migrates from effortful recall to automatic habit — the hallmark of "
        "consolidated procedural memory."
    ),
    "vocab": (
        "**Why this works — elaborative encoding.** Using a new word in your own "
        "sentence — rather than just reading it — connects it to your existing semantic "
        "network. The richer the connection, the stronger the trace. Words you *produce* "
        "become words you can reach for mid-sentence."
    ),
}


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------

def load_progress() -> list[dict]:
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_result(drill: str, scores: dict, overall: float):
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    history = load_progress()
    history.append({
        "date": datetime.date.today().isoformat(),
        "drill": drill,
        "scores": scores,
        "overall": round(overall, 1),
    })
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def compute_streak(history: list[dict]) -> int:
    """Consecutive days (ending today or yesterday) with at least one drill."""
    days = {datetime.date.fromisoformat(h["date"]) for h in history}
    if not days:
        return 0
    today = datetime.date.today()
    start = today if today in days else today - datetime.timedelta(days=1)
    if start not in days:
        return 0
    streak = 0
    day = start
    while day in days:
        streak += 1
        day -= datetime.timedelta(days=1)
    return streak


def skill_averages(history: list[dict]) -> dict[str, float]:
    totals: dict[str, list[float]] = {s: [] for s in SKILLS}
    for h in history:
        for skill, value in (h.get("scores") or {}).items():
            if skill in totals and isinstance(value, (int, float)):
                totals[skill].append(value)
    return {s: round(sum(v) / len(v), 1) for s, v in totals.items() if v}


def weakest_skill(history: list[dict]) -> str | None:
    avgs = skill_averages(history)
    if not avgs:
        return None
    return min(avgs, key=avgs.get)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def wav_duration_seconds(audio_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def transcribe_audio(client, audio_file) -> str | None:
    """Transcribe recorded audio verbatim (keeping filler words)."""
    try:
        audio_file.seek(0)
        result = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=("recording.wav", audio_file.read(), "audio/wav"),
            prompt="Transcribe verbatim, including filler words like um, uh, like, you know.",
        )
        return result.text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None


def twister_accuracy(target: str, spoken: str) -> float:
    """Word-level similarity between target twister and what was actually said."""
    normalize = lambda s: [w.strip(".,;:!?'\"").lower() for w in s.split() if w.strip(".,;:!?'\"")]
    return round(difflib.SequenceMatcher(None, normalize(target), normalize(spoken)).ratio() * 100, 1)


FILLER_WORDS = {"um", "uh", "erm", "like", "basically", "actually", "literally", "right", "so", "well"}


def count_fillers(text: str) -> int:
    words = [w.strip(".,;:!?").lower() for w in text.split()]
    return sum(1 for w in words if w in FILLER_WORDS)


# ---------------------------------------------------------------------------
# AI feedback
# ---------------------------------------------------------------------------

FEEDBACK_SYSTEM = """You are an elite executive communication coach with decades of experience training leaders, TED speakers, and senior executives. You are coaching a client through a communication drill. Speak in first person, warm but direct — like a trusted mentor who tells the truth kindly.

You MUST respond with a single JSON object with exactly these keys:
- "scores": object with integer scores 1-10 for "clarity", "concision", "structure", "impact"
- "strengths": array of 1-3 short strings — what genuinely worked
- "improvements": array of 1-3 short strings — the highest-leverage fixes
- "upgraded_version": string — your rewritten, stronger version of their response (same length or shorter)
- "coach_note": string — 1-2 encouraging sentences in your voice, referencing neuroplasticity when natural (e.g. "every rep is wiring this in")

Score honestly. A rambling answer should score low on concision. Reserve 9-10 for genuinely executive-level responses."""


def get_ai_feedback(client, drill_name: str, task: str, response_text: str,
                    extra_context: str = "") -> dict | None:
    user_msg = (
        f"DRILL: {drill_name}\n\n"
        f"TASK GIVEN TO CLIENT:\n{task}\n\n"
        f"CLIENT'S RESPONSE:\n{response_text}\n"
    )
    if extra_context:
        user_msg += f"\nADDITIONAL CONTEXT:\n{extra_context}\n"

    try:
        resp = client.chat.completions.create(
            model=FEEDBACK_MODEL,
            messages=[
                {"role": "system", "content": FEEDBACK_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=900,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"Feedback generation failed: {e}")
        return None


def render_feedback(feedback: dict, drill_key: str):
    scores = feedback.get("scores", {})
    cols = st.columns(4)
    for col, skill in zip(cols, SKILLS):
        with col:
            st.metric(skill.capitalize(), f"{scores.get(skill, '—')}/10")

    overall = None
    numeric = [v for v in scores.values() if isinstance(v, (int, float))]
    if numeric:
        overall = sum(numeric) / len(numeric)

    if feedback.get("strengths"):
        st.markdown("**What worked**")
        for s in feedback["strengths"]:
            st.markdown(f"- ✅ {s}")
    if feedback.get("improvements"):
        st.markdown("**Level up**")
        for s in feedback["improvements"]:
            st.markdown(f"- 🎯 {s}")
    if feedback.get("upgraded_version"):
        st.markdown("**How I'd say it**")
        st.info(feedback["upgraded_version"])
    if feedback.get("coach_note"):
        st.markdown(f"> 💬 *{feedback['coach_note']}*")

    if overall is not None:
        save_result(drill_key, scores, overall)


# ---------------------------------------------------------------------------
# Daily deterministic picks (same exercises all day → spaced variation day to day)
# ---------------------------------------------------------------------------

def daily_rng(salt: str) -> random.Random:
    return random.Random(f"{datetime.date.today().isoformat()}-{salt}")


def pick_exercise(key: str, bank_size: int, salt: str) -> int:
    if key not in st.session_state:
        st.session_state[key] = daily_rng(salt).randrange(bank_size)
    return st.session_state[key]


def new_exercise_button(key: str, bank_size: int, label: str = "🔄 New exercise"):
    if st.button(label, key=f"{key}_new"):
        current = st.session_state.get(key, 0)
        choices = [i for i in range(bank_size) if i != current]
        st.session_state[key] = random.choice(choices) if choices else current
        st.rerun()


# ---------------------------------------------------------------------------
# Drill tabs
# ---------------------------------------------------------------------------

def render_warmup_tab(client):
    st.markdown("Loosen up the articulators. Say the twister below **three times, fast and clean**, then record your best run.")
    with st.expander("🧠 Why this works"):
        st.markdown(NEURO_NOTES["warmup"])

    idx = pick_exercise("at_warmup_idx", len(TONGUE_TWISTERS), "warmup")
    twister, difficulty = TONGUE_TWISTERS[idx]
    st.info(f"**{twister}**  \n*Difficulty: {difficulty}*")
    new_exercise_button("at_warmup_idx", len(TONGUE_TWISTERS))

    audio = st.audio_input("Record your best run", key="at_warmup_audio")
    if audio and st.button("Score my diction", key="at_warmup_go", type="primary"):
        with st.spinner("Listening closely..."):
            transcript = transcribe_audio(client, audio)
        if transcript:
            accuracy = twister_accuracy(twister, transcript)
            col1, col2 = st.columns(2)
            col1.metric("Diction accuracy", f"{accuracy}%")
            duration = wav_duration_seconds(audio.getvalue())
            if duration:
                wpm = round(len(transcript.split()) / duration * 60)
                col2.metric("Pace", f"{wpm} wpm")
            st.caption(f"What I heard: *{transcript}*")
            if accuracy >= 90:
                st.success("Crisp! Your articulators are warm. Move on to the impromptu round.")
            elif accuracy >= 70:
                st.warning("Close — slow down 10% and hit every consonant. Precision first, speed second.")
            else:
                st.error("Muddy run. Say it once at half speed, exaggerating each syllable, then re-record.")
            save_result("warmup", {"clarity": min(10, round(accuracy / 10))}, accuracy / 10)


def render_impromptu_tab(client):
    st.markdown("You have a scenario and ~60 seconds. **Record yourself speaking** (best for training) or type your answer.")
    with st.expander("🧠 Why this works"):
        st.markdown(NEURO_NOTES["impromptu"])

    idx = pick_exercise("at_impromptu_idx", len(IMPROMPTU_TOPICS), "impromptu")
    topic = IMPROMPTU_TOPICS[idx]
    st.info(f"🎯 **{topic}**")
    new_exercise_button("at_impromptu_idx", len(IMPROMPTU_TOPICS))

    audio = st.audio_input("Record your answer (aim for 45–75 seconds)", key="at_impromptu_audio")
    typed = st.text_area("...or type it", key="at_impromptu_text", height=140,
                         placeholder="If you can't speak out loud right now, write exactly what you would say.")

    if st.button("Get coaching", key="at_impromptu_go", type="primary"):
        response_text, extra = None, ""
        if audio:
            with st.spinner("Transcribing..."):
                response_text = transcribe_audio(client, audio)
            if response_text:
                duration = wav_duration_seconds(audio.getvalue())
                fillers = count_fillers(response_text)
                if duration:
                    wpm = round(len(response_text.split()) / duration * 60)
                    extra = (f"This was SPOKEN. Duration: {round(duration)}s, pace: {wpm} wpm "
                             f"(conversational ideal is 130-160), filler words detected: {fillers}. "
                             f"Comment on pace and fillers.")
                c1, c2 = st.columns(2)
                if duration:
                    c1.metric("Pace", f"{wpm} wpm")
                c2.metric("Filler words", fillers)
        elif typed.strip():
            response_text = typed.strip()
            extra = "This was TYPED (client couldn't speak aloud). Skip pace/filler comments."

        if not response_text:
            st.warning("Record or type an answer first.")
        else:
            with st.expander("Your transcript"):
                st.write(response_text)
            with st.spinner("Your coach is thinking..."):
                feedback = get_ai_feedback(client, "Impromptu speaking (60 seconds)", topic, response_text, extra)
            if feedback:
                render_feedback(feedback, "impromptu")


def render_precision_tab(client):
    st.markdown("Below is a bloated passage. **Rewrite it in as few words as possible** without losing meaning. Target: cut it by at least half.")
    with st.expander("🧠 Why this works"):
        st.markdown(NEURO_NOTES["precision"])

    idx = pick_exercise("at_precision_idx", len(CONCISION_CHALLENGES), "precision")
    passage = CONCISION_CHALLENGES[idx]
    st.info(passage)
    st.caption(f"Original: {len(passage.split())} words")
    new_exercise_button("at_precision_idx", len(CONCISION_CHALLENGES))

    rewrite = st.text_area("Your tightened version", key="at_precision_text", height=100)
    if st.button("Score my rewrite", key="at_precision_go", type="primary"):
        if not rewrite.strip():
            st.warning("Write your rewrite first.")
        else:
            original_words = len(passage.split())
            new_words = len(rewrite.split())
            reduction = round((1 - new_words / original_words) * 100)
            c1, c2 = st.columns(2)
            c1.metric("Your version", f"{new_words} words")
            c2.metric("Reduction", f"{reduction}%")
            with st.spinner("Your coach is thinking..."):
                feedback = get_ai_feedback(
                    client, "Concision rewrite",
                    f"Rewrite this passage as concisely as possible without losing meaning:\n{passage}",
                    rewrite.strip(),
                    f"Original was {original_words} words; client used {new_words} ({reduction}% reduction). "
                    f"Check no essential meaning was lost.",
                )
            if feedback:
                render_feedback(feedback, "precision")


def render_structure_tab(client):
    st.markdown("Pick a framework, get a prompt, and answer **using that structure explicitly**.")
    with st.expander("🧠 Why this works"):
        st.markdown(NEURO_NOTES["structure"])

    framework_name = st.selectbox("Framework", list(FRAMEWORKS.keys()), key="at_framework_sel")
    fw = FRAMEWORKS[framework_name]
    st.markdown(f"{fw['structure']}  \n*{fw['description']}*")

    prompts = fw["prompts"]
    idx_key = f"at_structure_idx_{framework_name}"
    idx = pick_exercise(idx_key, len(prompts), f"structure-{framework_name}")
    prompt = prompts[idx]
    st.info(f"🎯 **{prompt}**")
    new_exercise_button(idx_key, len(prompts), "🔄 New prompt")

    audio = st.audio_input("Record your structured answer", key=f"at_structure_audio_{framework_name}")
    typed = st.text_area("...or type it", key=f"at_structure_text_{framework_name}", height=140)

    if st.button("Check my structure", key="at_structure_go", type="primary"):
        response_text = None
        if audio:
            with st.spinner("Transcribing..."):
                response_text = transcribe_audio(client, audio)
        elif typed.strip():
            response_text = typed.strip()

        if not response_text:
            st.warning("Record or type an answer first.")
        else:
            with st.expander("Your transcript"):
                st.write(response_text)
            with st.spinner("Your coach is thinking..."):
                feedback = get_ai_feedback(
                    client, f"Structured answer using {framework_name}",
                    f"{prompt}\n(Required structure: {fw['structure']})",
                    response_text,
                    f"Grade primarily on whether each element of {framework_name} is clearly present and in order.",
                )
            if feedback:
                render_feedback(feedback, "structure")


def render_vocab_tab(client):
    st.markdown("Today's power word. **Use it in a sentence you'd actually say at work** — natural, not forced.")
    with st.expander("🧠 Why this works"):
        st.markdown(NEURO_NOTES["vocab"])

    idx = pick_exercise("at_vocab_idx", len(VOCAB_WORDS), "vocab")
    word, definition = VOCAB_WORDS[idx]
    st.info(f"**{word}** — *{definition}*")
    new_exercise_button("at_vocab_idx", len(VOCAB_WORDS), "🔄 New word")

    sentence = st.text_input("Your sentence", key="at_vocab_text",
                             placeholder=f"Write a work sentence using “{word}”...")
    if st.button("Check my sentence", key="at_vocab_go", type="primary"):
        if not sentence.strip():
            st.warning("Write a sentence first.")
        elif word.lower() not in sentence.lower():
            st.warning(f"Your sentence needs to actually use “{word}”.")
        else:
            with st.spinner("Your coach is thinking..."):
                feedback = get_ai_feedback(
                    client, "Vocabulary in context",
                    f"Use the word “{word}” ({definition}) naturally in a professional sentence.",
                    sentence.strip(),
                    "Judge whether the word is used correctly and sounds natural, not forced.",
                )
            if feedback:
                render_feedback(feedback, "vocab")


def render_progress_tab():
    history = load_progress()
    if not history:
        st.markdown("No drills completed yet. Your progress will show up here — **consistency beats intensity**, so aim for one drill a day.")
        return

    streak = compute_streak(history)
    avgs = skill_averages(history)
    overall_avg = round(sum(h["overall"] for h in history) / len(history), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 Streak", f"{streak} day{'s' if streak != 1 else ''}")
    c2.metric("Drills done", len(history))
    c3.metric("Avg score", f"{overall_avg}/10")
    weak = weakest_skill(history)
    c4.metric("Focus area", weak.capitalize() if weak else "—")

    if avgs:
        st.markdown("**Skill breakdown**")
        cols = st.columns(len(avgs))
        for col, (skill, avg) in zip(cols, avgs.items()):
            col.metric(skill.capitalize(), f"{avg}/10")

    try:
        import pandas as pd
        df = pd.DataFrame(history)
        daily = df.groupby("date")["overall"].mean().rename("Average score")
        if len(daily) > 1:
            st.markdown("**Score over time**")
            st.line_chart(daily)
    except Exception:
        pass

    if weak:
        tab_for_skill = {"clarity": "Warm-Up and Impromptu", "concision": "Precision",
                         "structure": "Structure", "impact": "Impromptu"}
        st.markdown(
            f"> 💬 *Your lowest average is **{weak}**. Neuroplasticity is use-dependent — "
            f"the circuit you train is the circuit that grows. Spend extra reps in the "
            f"**{tab_for_skill.get(weak, 'Impromptu')}** drill this week.*"
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def render_trainer_sidebar():
    """Compact stats for the sidebar while in trainer mode."""
    history = load_progress()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Streak", f"{compute_streak(history)} days")
    with col2:
        st.metric("Drills", f"{len(history)}")
    weak = weakest_skill(history)
    if weak:
        st.caption(f"Focus area: **{weak.capitalize()}**")


def render_trainer(client):
    st.markdown('<p class="hero-heading">Neuroplasticity &amp; Articulation Trainer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Five minutes a day rewires how you speak. Warm up your voice, '
        'think on your feet, cut the fluff, and build structures that stick — with instant '
        'coaching on every rep.</p>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["🔥 Warm-Up", "🎤 Impromptu", "✂️ Precision", "🏗️ Structure", "📚 Vocabulary", "📈 Progress"])
    with tabs[0]:
        render_warmup_tab(client)
    with tabs[1]:
        render_impromptu_tab(client)
    with tabs[2]:
        render_precision_tab(client)
    with tabs[3]:
        render_structure_tab(client)
    with tabs[4]:
        render_vocab_tab(client)
    with tabs[5]:
        render_progress_tab()
