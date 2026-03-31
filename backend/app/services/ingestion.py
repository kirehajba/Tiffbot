import json
import logging

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import Video, async_session_factory
from app.services.youtube import YouTubeService

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.youtube = YouTubeService()
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.chroma_client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="video_transcripts",
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def ingest_channel(self) -> int:
        """Fetch all channel videos and ingest their transcripts."""
        videos = await self.youtube.fetch_channel_videos()
        logger.info(f"Found {len(videos)} videos to process")

        async with async_session_factory() as db:
            for video_data in videos:
                existing = await db.execute(
                    select(Video).where(Video.youtube_id == video_data["youtube_id"])
                )
                if existing.scalar_one_or_none():
                    continue

                video = Video(
                    youtube_id=video_data["youtube_id"],
                    title=video_data["title"],
                    description=video_data.get("description"),
                    thumbnail_url=video_data.get("thumbnail_url"),
                    channel_title=video_data.get("channel_title"),
                    published_at=video_data.get("published_at"),
                    transcript_status="pending",
                )
                db.add(video)

            await db.commit()

        async with async_session_factory() as db:
            result = await db.execute(
                select(Video).where(Video.transcript_status == "pending")
            )
            pending_videos = result.scalars().all()

        processed = 0
        for video in pending_videos:
            try:
                await self._process_video(video)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process video {video.youtube_id}: {e}")
                async with async_session_factory() as db:
                    result = await db.execute(
                        select(Video).where(Video.id == video.id)
                    )
                    v = result.scalar_one()
                    v.transcript_status = "failed"
                    await db.commit()

        return processed

    async def _process_video(self, video: Video):
        """Process a single video: get transcript, chunk, embed, store."""
        segments = await self.youtube.get_transcript(video.youtube_id)
        if not segments:
            async with async_session_factory() as db:
                result = await db.execute(select(Video).where(Video.id == video.id))
                v = result.scalar_one()
                v.transcript_status = "failed"
                await db.commit()
            return

        blocks = self.youtube.build_full_text_with_timestamps(segments)

        chunks = []
        for block in blocks:
            sub_chunks = self.splitter.split_text(block["text"])
            for chunk_text in sub_chunks:
                chunks.append({
                    "text": chunk_text,
                    "start_time": block["start_time"],
                    "end_time": block["end_time"],
                })

        if not chunks:
            return

        batch_size = 100
        total_stored = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self._embed_texts(texts)

            ids = [f"{video.youtube_id}_{i + j}" for j in range(len(batch))]
            metadatas = [
                {
                    "video_id": video.id,
                    "youtube_id": video.youtube_id,
                    "title": video.title,
                    "thumbnail_url": video.thumbnail_url or "",
                    "start_time": c["start_time"],
                    "end_time": c["end_time"],
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_stored += len(batch)

        async with async_session_factory() as db:
            result = await db.execute(select(Video).where(Video.id == video.id))
            v = result.scalar_one()
            v.transcript_status = "completed"
            v.chunk_count = total_stored
            await db.commit()

        logger.info(f"Ingested {total_stored} chunks for video: {video.title}")

    async def get_status(self) -> dict:
        async with async_session_factory() as db:
            total = (await db.execute(select(func.count(Video.id)))).scalar() or 0
            completed = (
                await db.execute(
                    select(func.count(Video.id)).where(Video.transcript_status == "completed")
                )
            ).scalar() or 0
            failed = (
                await db.execute(
                    select(func.count(Video.id)).where(Video.transcript_status == "failed")
                )
            ).scalar() or 0
            pending = (
                await db.execute(
                    select(func.count(Video.id)).where(Video.transcript_status == "pending")
                )
            ).scalar() or 0
            total_chunks = (
                await db.execute(select(func.sum(Video.chunk_count)))
            ).scalar() or 0

        return {
            "total_videos": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "total_chunks": total_chunks,
        }
