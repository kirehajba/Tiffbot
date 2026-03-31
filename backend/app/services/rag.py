import logging
from typing import AsyncGenerator

import chromadb
from openai import OpenAI

from app.core.config import settings
from app.services.llm import stream_response

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.chroma_client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="video_transcripts",
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_query(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most relevant transcript chunks for a query."""
        query_embedding = self._embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                chunks.append({
                    "text": doc,
                    "video_id": metadata.get("video_id", ""),
                    "youtube_id": metadata.get("youtube_id", ""),
                    "title": metadata.get("title", ""),
                    "thumbnail_url": metadata.get("thumbnail_url", ""),
                    "start_time": metadata.get("start_time", 0),
                    "end_time": metadata.get("end_time", 0),
                    "relevance_score": 1 - distance,
                })

        return chunks

    async def query_stream(
        self,
        question: str,
        chat_history: list[dict],
        provider: str = "openai",
        top_k: int = 5,
    ) -> tuple[AsyncGenerator[str, None], list[dict]]:
        """Run RAG: retrieve context, then stream LLM response. Returns (stream, sources)."""
        chunks = self.retrieve(question, top_k=top_k)

        sources = []
        seen_videos = set()
        for chunk in chunks:
            vid = chunk["youtube_id"]
            if vid not in seen_videos:
                seen_videos.add(vid)
                sources.append({
                    "video_id": chunk["video_id"],
                    "youtube_id": chunk["youtube_id"],
                    "title": chunk["title"],
                    "thumbnail_url": chunk["thumbnail_url"],
                    "timestamp_seconds": chunk["start_time"],
                    "relevance_score": chunk["relevance_score"],
                })

        token_stream = stream_response(provider, chunks, question, chat_history)
        return token_stream, sources
