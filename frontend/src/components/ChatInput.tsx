"use client";

import { useState, useRef, useEffect } from "react";
import { SendHorizontal } from "lucide-react";
import { cn } from "@/lib/cn";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  provider: string;
  onProviderChange: (provider: string) => void;
}

export default function ChatInput({
  onSend,
  disabled,
  provider,
  onProviderChange,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 200) + "px";
    }
  }, [input]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-3">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-xl border border-gray-300 bg-white px-4 py-2 shadow-sm transition focus-within:border-brand-400 focus-within:ring-2 focus-within:ring-brand-100">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your business coaching question..."
            disabled={disabled}
            rows={1}
            className="max-h-[200px] flex-1 resize-none bg-transparent py-1.5 text-sm outline-none placeholder:text-gray-400"
          />
          <button
            onClick={handleSubmit}
            disabled={disabled || !input.trim()}
            className={cn(
              "shrink-0 rounded-lg p-2 transition",
              input.trim() && !disabled
                ? "bg-brand-600 text-white hover:bg-brand-700"
                : "bg-gray-100 text-gray-400"
            )}
          >
            <SendHorizontal className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Model:</span>
            <select
              value={provider}
              onChange={(e) => onProviderChange(e.target.value)}
              className="rounded border border-gray-200 px-2 py-0.5 text-xs text-gray-600 outline-none focus:border-brand-400"
            >
              <option value="openai">GPT-4o</option>
              <option value="anthropic">Claude</option>
            </select>
          </div>
          <p className="text-xs text-gray-400">
            Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}
