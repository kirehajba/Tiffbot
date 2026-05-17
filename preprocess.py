"""
Pre-process and ingest all transcripts into ChromaDB.

Run this BEFORE starting the Streamlit app:
    python preprocess.py && streamlit run demo.py
"""

from __future__ import annotations

import glob
import os
import sys

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chroma_data")
YOUTUBE_TRANSCRIPT_DIR = os.path.join(os.path.dirname(__file__), "youtube_transcripts")
TRANSCRIPT_DIR = os.path.join(os.path.dirname(__file__), ".transcripts")


def main():
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY in your .env file first.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(
        name="video_transcripts",
        metadata={"hnsw:space": "cosine"},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Gather all transcript .txt files
    txt_files = []
    for folder in [YOUTUBE_TRANSCRIPT_DIR, TRANSCRIPT_DIR]:
        if os.path.isdir(folder):
            txt_files.extend(sorted(glob.glob(os.path.join(folder, "*.txt"))))

    if not txt_files:
        print("No transcript files found. Nothing to ingest.")
        return

    # Find already-indexed video IDs
    existing_ids = set()
    try:
        existing = collection.get()
        if existing and existing["metadatas"]:
            existing_ids = {m.get("video_id") for m in existing["metadatas"]}
    except Exception:
        pass

    print(f"Found {len(txt_files)} transcript files, {len(existing_ids)} already indexed.")

    ingested, skipped, failed = 0, 0, 0

    for i, txt_path in enumerate(txt_files, 1):
        filename = os.path.basename(txt_path)
        video_id = filename[:-4] if filename.endswith(".txt") else filename

        if video_id in existing_ids:
            skipped += 1
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if len(text) < 50:
            skipped += 1
            continue

        title = os.path.splitext(filename)[0]
        chunks = splitter.split_text(text)
        if not chunks:
            failed += 1
            continue

        print(f"[{i}/{len(txt_files)}] Embedding: {title[:70]}... ({len(chunks)} chunks)")

        for batch_start in range(0, len(chunks), 50):
            batch = chunks[batch_start : batch_start + 50]
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
            embeddings = [item.embedding for item in resp.data]
            ids = [f"{video_id}_{batch_start + j}" for j in range(len(batch))]
            metadatas = [
                {
                    "video_id": video_id,
                    "title": title,
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

    total = collection.count()
    print(f"\nDone! Ingested: {ingested}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total chunks in knowledge base: {total}")
    print("Ready — start the app with: streamlit run demo.py")


if __name__ == "__main__":
    main()
