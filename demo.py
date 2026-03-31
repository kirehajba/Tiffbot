"""
TiiffBot Demo — AI Business Coach powered by Tiffany Cheng's YouTube content.

Run:  streamlit run demo.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY", "")
    or st.secrets.get("OPENAI_API_KEY", "")
)
CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chroma_data")
TRANSCRIPT_DIR = os.path.join(os.path.dirname(__file__), ".transcripts")
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "video_training")

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You ARE Tiffany Cheng. Speak in first person ("I", "my", "me") at all times. You are an executive leadership coach who held two global director positions at Volvo Group (Belgium and Sweden), became Vice President of Brand Marketing and Corporate Communication at Volvo, and then Global Vice President of Communication at Atlas Copco. You have helped leaders secure roles as CFOs, CTOs, Partners, GMs, VPs, SVPs, Executive Directors, and Senior Directors.

You speak from your own lived experience. When you share advice, you say things like "When I was in that exact position..." or "What I tell my clients is..." or "In my experience...". You are warm, direct, and encouraging — like a trusted mentor who genuinely cares about the person's success.

IMPORTANT RULES:
1. ALWAYS speak as Tiffany in first person. Never say "Tiffany says" or "According to Tiffany" — YOU are Tiffany.
2. Ground your answers in the provided video transcript context. Draw on it naturally as if recalling your own thoughts and experiences.
3. If the context doesn't cover the topic well enough, be honest: "I haven't covered that in depth yet, but here's what I'd suggest..." and point them to inspiremyday.org or your free masterclass.
4. Be encouraging, practical, and actionable. Give concrete next steps when possible.
5. Use clear, confident language suitable for ambitious business professionals.
6. If asked about topics completely outside your expertise, gently redirect: "That's outside my area, but what I can help you with is..."
7. Occasionally reference which video the advice comes from naturally, e.g. "I talk about this in my video on executive resumes...".
8. Do NOT include follow-up question suggestions in your answer — those are handled separately."""


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name="video_transcripts",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Local video discovery + transcription
# ---------------------------------------------------------------------------

def get_ffmpeg_path() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _clean_title(raw: str) -> str:
    title = os.path.splitext(raw)[0]
    for suffix in ["(360p, h264)", "(720p, h264)", "(1080p, h264)", "(360p, h265)", "(1)"]:
        title = title.replace(suffix, "")
    title = title.replace(" - Tiffany Cheng", "").strip().rstrip("-").strip()
    if title.endswith(".mp4"):
        title = title[:-4].strip()
    return title


def discover_videos() -> list[dict]:
    """Find videos from mp4 files or cached transcripts."""
    seen_ids = set()
    videos = []

    # First: discover from actual video files
    if os.path.isdir(VIDEO_DIR):
        for filepath in sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))):
            filename = os.path.basename(filepath)
            videos.append({
                "filepath": filepath,
                "filename": filename,
                "title": _clean_title(filename),
                "video_id": filename,
            })
            seen_ids.add(filename)

    # Second: discover from cached transcripts (for cloud deployment without video files)
    if os.path.isdir(TRANSCRIPT_DIR):
        for txt_path in sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt"))):
            txt_name = os.path.basename(txt_path)
            video_id = txt_name[:-4] if txt_name.endswith(".txt") else txt_name
            if video_id not in seen_ids:
                videos.append({
                    "filepath": None,
                    "filename": video_id,
                    "title": _clean_title(video_id),
                    "video_id": video_id,
                })
                seen_ids.add(video_id)

    return videos


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from video as mp3 using bundled ffmpeg."""
    ffmpeg = get_ffmpeg_path()
    try:
        subprocess.run(
            [ffmpeg, "-i", video_path, "-vn", "-acodec", "libmp3lame",
             "-ab", "64k", "-ar", "16000", "-ac", "1", "-y", output_path],
            capture_output=True, check=True, timeout=120,
        )
        return True
    except Exception as e:
        st.warning(f"Audio extraction failed for {os.path.basename(video_path)}: {e}")
        return False


def transcribe_video(video: dict) -> str | None:
    """Transcribe a video file. Returns full transcript text, cached locally."""
    cache_file = os.path.join(TRANSCRIPT_DIR, video["filename"] + ".txt")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()

    if not video.get("filepath") or not os.path.exists(video["filepath"]):
        return None

    client = get_openai_client()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not extract_audio(video["filepath"], tmp_path):
            return None

        file_size = os.path.getsize(tmp_path)
        if file_size > 24 * 1024 * 1024:
            # Split into ~20MB chunks for very long videos
            return _transcribe_large_audio(tmp_path, video, cache_file)

        with open(tmp_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )

        with open(cache_file, "w") as f:
            f.write(result)

        return result

    except Exception as e:
        st.warning(f"Transcription failed for {video['title']}: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _transcribe_large_audio(audio_path: str, video: dict, cache_file: str) -> str | None:
    """Split a large audio file into segments and transcribe each."""
    ffmpeg = get_ffmpeg_path()
    client = get_openai_client()
    parts = []
    segment_dur = 600  # 10-minute segments

    # Get duration
    try:
        probe = subprocess.run(
            [ffmpeg, "-i", audio_path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        # Parse duration from ffmpeg stderr
        import re
        match = re.search(r"Duration: (\d+):(\d+):(\d+)", probe.stderr)
        if match:
            total_secs = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
        else:
            total_secs = 1800  # default 30 min
    except Exception:
        total_secs = 1800

    for start in range(0, total_secs, segment_dur):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            seg_path = tmp.name

        try:
            subprocess.run(
                [ffmpeg, "-i", audio_path, "-ss", str(start), "-t", str(segment_dur),
                 "-acodec", "libmp3lame", "-ab", "64k", "-ar", "16000", "-ac", "1", "-y", seg_path],
                capture_output=True, check=True, timeout=60,
            )

            with open(seg_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="text",
                )
            parts.append(result)
        except Exception as e:
            st.warning(f"Segment transcription failed: {e}")
        finally:
            if os.path.exists(seg_path):
                os.unlink(seg_path)

    if not parts:
        return None

    full_text = "\n\n".join(parts)
    with open(cache_file, "w") as f:
        f.write(full_text)

    return full_text


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_videos(progress_bar, status_text):
    collection = get_chroma_collection()
    client = get_openai_client()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
    )

    videos = discover_videos()
    total = len(videos)
    if total == 0:
        status_text.text("No videos found in video_training/ folder.")
        return {"ingested": 0, "skipped": 0, "failed": 0, "total": 0}

    existing_ids = set()
    try:
        existing = collection.get()
        if existing and existing["metadatas"]:
            existing_ids = {m.get("video_id") for m in existing["metadatas"]}
    except Exception:
        pass

    ingested, skipped, failed = 0, 0, 0

    for i, video in enumerate(videos):
        progress_bar.progress(
            (i + 1) / total,
            text=f"Processing {i+1}/{total}: {video['title'][:60]}",
        )

        if video["video_id"] in existing_ids:
            skipped += 1
            continue

        status_text.text(f"Transcribing: {video['title'][:60]}...")
        transcript = transcribe_video(video)
        if not transcript:
            failed += 1
            continue

        status_text.text(f"Chunking & embedding: {video['title'][:50]}...")
        chunks = splitter.split_text(transcript)
        if not chunks:
            failed += 1
            continue

        for batch_start in range(0, len(chunks), 50):
            batch = chunks[batch_start : batch_start + 50]
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
            embeddings = [item.embedding for item in resp.data]
            ids = [f"{video['video_id']}_{batch_start + j}" for j in range(len(batch))]
            metadatas = [
                {
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "thumbnail_url": "",
                }
                for _ in batch
            ]
            collection.upsert(
                ids=ids, documents=batch, embeddings=embeddings, metadatas=metadatas,
            )

        ingested += 1

    progress_bar.progress(1.0, text="Done!")
    return {"ingested": ingested, "skipped": skipped, "failed": failed, "total": total}


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    client = get_openai_client()
    collection = get_chroma_collection()

    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_emb = resp.data[0].embedding

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            chunks.append({
                "text": doc,
                "title": meta.get("title", ""),
                "relevance": round(1 - dist, 3),
            })
    return chunks


def build_context_prompt(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f'[From: "{c["title"]}"]\n{c["text"]}')
    return "\n\n---\n\n".join(parts)


def dedupe_sources(chunks: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for c in chunks:
        title = c["title"]
        if title not in seen:
            seen.add(title)
            sources.append({"title": title, "relevance": c["relevance"]})
    return sources


def generate_followups(question: str, answer: str) -> list[str]:
    """Ask the LLM for 3 short follow-up questions based on the conversation."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Based on this Q&A between a user and an executive coach, suggest exactly 3 short, "
                f"specific follow-up questions the user might want to ask next. "
                f"Return ONLY the 3 questions, one per line, no numbering, no bullets, no extra text.\n\n"
                f"User asked: {question}\n\nCoach answered: {answer[:500]}"
            ),
        }],
        temperature=0.7,
        max_tokens=150,
    )
    lines = resp.choices[0].message.content.strip().split("\n")
    return [line.strip() for line in lines if line.strip()][:3]


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

LOGO_WHITE = "https://inspiremyday.org/wp-content/uploads/2026/01/INSPIRE-MY-DAY-LOGO-White.png"
LOGO_BLUE = "https://inspiremyday.org/wp-content/uploads/2026/01/Inspiremyday-blue-logo.png"

st.set_page_config(
    page_title="Inspire My Day — AI Executive Coach",
    page_icon="https://inspiremyday.org/wp-content/uploads/2026/01/Inspiremyday-blue-logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

:root {
    --navy:       #1B2A4A;
    --navy-light: #243556;
    --gold:       #C9A84C;
    --gold-light: #E8D48B;
    --cream:      #FAF8F5;
    --slate:      #64748B;
    --white:      #FFFFFF;
}

/* Global */
.stApp {
    background-color: var(--cream) !important;
}
.main .block-container {
    max-width: 860px;
    padding-top: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2A4A 0%, #162240 100%) !important;
    border-right: 1px solid rgba(201,168,76,0.2);
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(201,168,76,0.25) !important;
}
section[data-testid="stSidebar"] .stMetric label {
    color: #94A3B8 !important;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #C9A84C !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(201,168,76,0.4) !important;
    color: #C9A84C !important;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.82rem;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(201,168,76,0.12) !important;
    border-color: #C9A84C !important;
}

/* Chat messages */
.stChatMessage {
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stChatMessageContent"] p {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #1E293B;
}

/* Chat input */
.stChatInput {
    border-color: #CBD5E1 !important;
}
.stChatInput > div {
    border-radius: 12px !important;
    border: 1.5px solid #CBD5E1 !important;
    box-shadow: 0 1px 4px rgba(27,42,74,0.06) !important;
}
.stChatInput > div:focus-within {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}

/* Follow-up / suggestion buttons */
.main .stButton > button {
    background: var(--white) !important;
    border: 1.5px solid #E2E8F0 !important;
    color: #1B2A4A !important;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.84rem;
    font-weight: 500;
    padding: 0.6rem 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    text-align: left !important;
}
.main .stButton > button:hover {
    border-color: #C9A84C !important;
    color: #C9A84C !important;
    box-shadow: 0 2px 8px rgba(201,168,76,0.12);
    transform: translateY(-1px);
}

/* Hero heading */
.hero-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 600;
    color: #1B2A4A;
    text-align: center;
    margin-bottom: 0.25rem;
    line-height: 1.3;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #64748B;
    text-align: center;
    margin-bottom: 2rem;
    line-height: 1.6;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
}
.hero-logo {
    display: block;
    margin: 0 auto 1.25rem auto;
    width: 200px;
}
.hero-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #C9A84C, #E8D48B);
    margin: 0 auto 1.5rem auto;
    border-radius: 2px;
}

/* Source chips */
.source-chip {
    display: inline-block;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    margin: 0.15rem 0.25rem;
    font-size: 0.78rem;
    color: #475569;
    font-family: 'Inter', sans-serif;
}

/* Sidebar video list */
.sidebar-video-item {
    font-size: 0.82rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    color: #94A3B8;
}

/* Section label */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

/* Followup section label */
.followup-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #94A3B8;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* Hide default Streamlit header/footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---- Sidebar ----
with st.sidebar:
    st.image(LOGO_WHITE, width=180)
    st.markdown("")
    st.caption("AI-Powered Executive Coaching")
    st.divider()

    collection = get_chroma_collection()
    try:
        count = collection.count()
    except Exception:
        count = 0

    local_videos = discover_videos()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Knowledge Base", f"{count} chunks")
    with col2:
        st.metric("Videos", f"{len(local_videos)}")

    if count == 0 and OPENAI_API_KEY and len(local_videos) > 0:
        st.info("Preparing knowledge base...")
        bar = st.progress(0, text="Starting...")
        status = st.empty()
        result = ingest_videos(bar, status)
        status.text("")
        st.success(
            f"Ready! Ingested **{result['ingested']}** videos, "
            f"skipped {result['skipped']}, failed {result['failed']}."
        )
        st.rerun()
    elif count == 0 and not OPENAI_API_KEY:
        st.error("Set OPENAI_API_KEY in your .env file (local) or Streamlit Secrets (cloud).")
    elif count == 0 and len(local_videos) == 0:
        st.warning("No videos or transcripts found.")

    st.divider()

    if local_videos:
        st.markdown("**Trained On**")
        for v in local_videos:
            st.markdown(
                f'<div class="sidebar-video-item">&#9654;&ensp;{v["title"]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

    st.divider()

    link_col1, link_col2 = st.columns(2)
    with link_col1:
        st.markdown("[Website](https://inspiremyday.org)")
    with link_col2:
        st.markdown("[YouTube](https://www.youtube.com/@inspiremydaytiffany)")

    st.markdown("")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Re-index", use_container_width=True):
            if not OPENAI_API_KEY:
                st.error("Set OPENAI_API_KEY first.")
            else:
                bar = st.progress(0, text="Starting...")
                status = st.empty()
                result = ingest_videos(bar, status)
                status.text("")
                st.success(f"Indexed {result['ingested']} videos.")
                st.rerun()
    with btn_col2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("")
    st.caption("Powered by Inspire My Day")


# ---- Main chat area ----

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg_idx, msg in enumerate(st.session_state.messages):
    avatar = LOGO_BLUE if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips_html = "".join(
                f'<span class="source-chip">&#9654; {src["title"]}</span>'
                for src in msg["sources"]
            )
            st.markdown(chips_html, unsafe_allow_html=True)

# Show follow-up buttons only on the LAST assistant message
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
    and st.session_state.messages[-1].get("followups")
):
    st.markdown('<p class="followup-label">Continue the conversation</p>', unsafe_allow_html=True)
    followups = st.session_state.messages[-1]["followups"]
    cols = st.columns(len(followups))
    for i, q in enumerate(followups):
        with cols[i]:
            if st.button(q, use_container_width=True, key=f"followup_{i}"):
                st.session_state.pending_prompt = q
                st.rerun()

# Empty / welcome state
if not st.session_state.messages and count > 0:
    st.markdown("")
    st.markdown(
        f'<img src="{LOGO_BLUE}" class="hero-logo" alt="Inspire My Day">',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-heading">Your AI Executive Coach</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">'
        "Hi, I'm Tiffany. I help high-performing directors and senior leaders "
        "advance into VP, GM, and C-suite roles. Ask me anything about your career."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-label" style="text-align:center;">Popular questions</p>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "Is the promotion from Director to VP really worth it?",
        "How should I write my executive resume?",
        "How do I build executive presence?",
        "Why do I feel underappreciated at work?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(q, use_container_width=True, key=f"suggestion_{i}"):
                st.session_state.pending_prompt = q
                st.rerun()

elif not st.session_state.messages and count == 0:
    st.markdown("")
    st.markdown(
        f'<img src="{LOGO_BLUE}" class="hero-logo" alt="Inspire My Day">',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-heading">Your AI Executive Coach</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Setting up the knowledge base. Please check the sidebar for status.</p>',
        unsafe_allow_html=True,
    )

# Chat input
prompt = st.session_state.pop("pending_prompt", None) or st.chat_input(
    "Ask about leadership, career growth, executive presence..."
)

if prompt:
    if not OPENAI_API_KEY:
        st.error("Please set your OPENAI_API_KEY in the .env file (local) or Streamlit Secrets (cloud).")
        st.stop()

    if count == 0:
        st.warning("Knowledge base is empty. Please wait for indexing to complete.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chunks = retrieve_context(prompt)
    context_text = build_context_prompt(chunks)
    sources = dedupe_sources(chunks)

    messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-6:]:
        messages_for_llm.append({"role": m["role"], "content": m["content"]})
    messages_for_llm.append({
        "role": "user",
        "content": (
            f"Based on the following video transcript excerpts, please answer the user's question.\n\n"
            f"VIDEO CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION: {prompt}"
        ),
    })

    with st.chat_message("assistant", avatar=LOGO_BLUE):
        client = get_openai_client()
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_for_llm,
            stream=True,
            temperature=0.7,
            max_tokens=1500,
        )

        def token_generator():
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token

        response = st.write_stream(token_generator())

        if sources:
            chips_html = "".join(
                f'<span class="source-chip">&#9654; {src["title"]}</span>'
                for src in sources
            )
            st.markdown(chips_html, unsafe_allow_html=True)

    followups = generate_followups(prompt, response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
        "followups": followups,
    })
    st.rerun()
