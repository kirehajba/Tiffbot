"""
Crawl all videos from the Inspire My Day YouTube channel
and save transcripts as plain text files.

Uses yt-dlp in batch mode with built-in rate limiting.

Usage:  python crawl_transcripts.py
"""

from __future__ import annotations

import glob
import os
import re
import subprocess

CHANNEL_URL = "https://www.youtube.com/@inspiremydaytiffany"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "youtube_transcripts")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_ffmpeg_dir() -> str:
    import imageio_ffmpeg
    return os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())


def vtt_to_text(vtt_path: str) -> str:
    """Strip VTT timestamps and metadata, return clean plain text."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = []
    seen = set()
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}", line):
            continue
        if re.match(r"^\d+$", line):
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)
    return " ".join(lines)


def main():
    ffmpeg_dir = get_ffmpeg_dir()
    env = os.environ.copy()
    env["PATH"] = ffmpeg_dir + ":" + env.get("PATH", "")

    vtt_dir = os.path.join(OUTPUT_DIR, "_vtt_raw")
    os.makedirs(vtt_dir, exist_ok=True)

    print(f"Downloading all subtitles from: {CHANNEL_URL}")
    print(f"Raw VTT files go to: {vtt_dir}")
    print(f"Clean text files go to: {OUTPUT_DIR}\n")

    result = subprocess.run(
        ["yt-dlp",
         "--cookies-from-browser", "chrome",
         "--write-auto-sub",
         "--sub-lang", "en",
         "--skip-download",
         "--sub-format", "vtt",
         "--sleep-subtitles", "3",
         "--sleep-requests", "1",
         "--no-overwrites",
         "-o", os.path.join(vtt_dir, "%(title)s [%(id)s].%(ext)s"),
         CHANNEL_URL],
        text=True, env=env, timeout=7200,
    )

    print(f"\nyt-dlp exited with code: {result.returncode}")

    vtt_files = glob.glob(os.path.join(vtt_dir, "*.vtt"))
    print(f"Found {len(vtt_files)} VTT files, converting to plain text...\n")

    converted = 0
    for vtt_path in sorted(vtt_files):
        basename = os.path.basename(vtt_path)
        name = re.sub(r"\s*\[[^\]]+\]\.en\.vtt$", "", basename)
        txt_name = re.sub(r'[\\/*?:"<>|｜]', "", name).strip()[:200] + ".txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_name)

        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 50:
            continue

        text = vtt_to_text(vtt_path)
        if len(text) > 50:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            converted += 1
            print(f"  Converted: {txt_name} ({len(text)} chars)")

    print(f"\nDone! Converted {converted} transcripts to plain text.")
    print(f"Transcripts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
