"use client";

import ReactMarkdown from "react-markdown";
import { Bot, User } from "lucide-react";
import { cn } from "@/lib/cn";
import VideoCard from "./VideoCard";
import type { VideoSource } from "@/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: VideoSource[];
  isStreaming?: boolean;
}

export default function ChatMessage({
  role,
  content,
  sources,
  isStreaming,
}: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <div
      className={cn("flex gap-4 px-4 py-6", isUser ? "bg-white" : "bg-gray-50")}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-brand-100 text-brand-700"
            : "bg-emerald-100 text-emerald-700"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div className="min-w-0 flex-1">
        <p className="mb-1 text-xs font-medium text-gray-500">
          {isUser ? "You" : "TiiffBot"}
        </p>

        <div className="prose prose-sm max-w-none text-gray-800">
          <ReactMarkdown>{content}</ReactMarkdown>
          {isStreaming && (
            <span className="inline-block h-4 w-1.5 animate-pulse bg-brand-500" />
          )}
        </div>

        {sources && sources.length > 0 && (
          <div className="mt-4">
            <p className="mb-2 text-xs font-medium text-gray-500">
              Sources from videos:
            </p>
            <div className="flex flex-wrap gap-2">
              {sources.map((source, i) => (
                <VideoCard key={i} source={source} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
