"use client";

import { ExternalLink } from "lucide-react";
import type { VideoSource } from "@/lib/api";

interface VideoCardProps {
  source: VideoSource;
}

export default function VideoCard({ source }: VideoCardProps) {
  const timestamp = Math.floor(source.timestamp_seconds || 0);
  const minutes = Math.floor(timestamp / 60);
  const seconds = timestamp % 60;
  const timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;
  const url = `https://www.youtube.com/watch?v=${source.youtube_id}&t=${timestamp}`;

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-center gap-3 rounded-lg border border-gray-200 bg-white p-2.5 transition hover:border-brand-300 hover:shadow-sm"
    >
      {source.thumbnail_url && (
        <img
          src={source.thumbnail_url}
          alt={source.title}
          className="h-12 w-20 rounded object-cover"
        />
      )}
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs font-medium text-gray-900 group-hover:text-brand-700">
          {source.title}
        </p>
        <p className="text-xs text-gray-500">at {timeStr}</p>
      </div>
      <ExternalLink className="h-3.5 w-3.5 shrink-0 text-gray-400 group-hover:text-brand-500" />
    </a>
  );
}
