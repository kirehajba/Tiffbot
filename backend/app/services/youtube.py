import logging
from typing import Optional

import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi

from app.core.config import settings

logger = logging.getLogger(__name__)


class YouTubeService:
    def __init__(self):
        self.channel_url = settings.youtube_channel_url

    def _extract_channel_handle(self) -> str:
        """Extract the @handle from the channel URL."""
        url = self.channel_url.rstrip("/")
        if "/@" in url:
            return url.split("/@")[-1]
        if "/channel/" in url:
            return url.split("/channel/")[-1]
        if "/c/" in url:
            return url.split("/c/")[-1]
        return url.split("/")[-1]

    async def fetch_channel_videos(self) -> list[dict]:
        """Fetch all videos from the channel using scrapetube (no API key needed)."""
        handle = self._extract_channel_handle()
        videos = []

        try:
            for video in scrapetube.get_channel(
                channel_url=self.channel_url, limit=None
            ):
                video_id = video["videoId"]
                title = video.get("title", {}).get("runs", [{}])[0].get("text", "Untitled")
                description_snippet = video.get("descriptionSnippet", {}).get("runs", [{}])[0].get("text", "")
                thumbnail_url = ""
                thumbnails = video.get("thumbnail", {}).get("thumbnails", [])
                if thumbnails:
                    thumbnail_url = thumbnails[-1].get("url", "")

                published_text = video.get("publishedTimeText", {}).get("simpleText", "")

                videos.append({
                    "youtube_id": video_id,
                    "title": title,
                    "description": description_snippet,
                    "thumbnail_url": thumbnail_url,
                    "channel_title": handle,
                    "published_at": published_text,
                })

            logger.info(f"Fetched {len(videos)} videos from @{handle}")
        except Exception as e:
            logger.error(f"Failed to fetch videos from @{handle}: {e}")

        return videos

    async def get_transcript(self, video_id: str, language: str = "en") -> Optional[list[dict]]:
        """Extract transcript for a single video. Returns list of segments with text, start, duration."""
        try:
            segments = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            return [
                {
                    "text": seg["text"],
                    "start": seg["start"],
                    "duration": seg["duration"],
                }
                for seg in segments
            ]
        except Exception as e:
            logger.warning(f"Could not get transcript for {video_id}: {e}")
            return None

    def build_full_text_with_timestamps(self, segments: list[dict]) -> list[dict]:
        """Group transcript segments into larger blocks preserving timestamp info."""
        if not segments:
            return []

        blocks = []
        current_text = ""
        current_start = segments[0]["start"]
        word_count = 0

        for seg in segments:
            current_text += " " + seg["text"]
            word_count += len(seg["text"].split())

            if word_count >= 150:
                blocks.append({
                    "text": current_text.strip(),
                    "start_time": current_start,
                    "end_time": seg["start"] + seg["duration"],
                })
                current_text = ""
                current_start = seg["start"] + seg["duration"]
                word_count = 0

        if current_text.strip():
            last_seg = segments[-1]
            blocks.append({
                "text": current_text.strip(),
                "start_time": current_start,
                "end_time": last_seg["start"] + last_seg["duration"],
            })

        return blocks
