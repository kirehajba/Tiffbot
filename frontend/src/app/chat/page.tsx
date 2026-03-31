"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import Sidebar from "@/components/Sidebar";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import {
  createSession,
  getSession,
  sendMessageStream,
  type VideoSource,
} from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: VideoSource[];
  isStreaming?: boolean;
}

export default function ChatPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [provider, setProvider] = useState("openai");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [user, loading, router]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const loadSession = useCallback(async (id: string) => {
    try {
      const data = await getSession(id);
      setSessionId(id);
      setMessages(
        data.messages.map((m) => ({
          id: m.id,
          role: m.role as "user" | "assistant",
          content: m.content,
          sources: m.sources ? JSON.parse(m.sources) : undefined,
        }))
      );
    } catch {
      /* ignore */
    }
  }, []);

  const handleNewChat = useCallback(() => {
    setSessionId(null);
    setMessages([]);
  }, []);

  const handleSend = async (content: string) => {
    let activeSession = sessionId;

    if (!activeSession) {
      try {
        const session = await createSession();
        activeSession = session.id;
        setSessionId(session.id);
      } catch {
        return;
      }
    }

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
    };
    const assistantMsg: Message = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsGenerating(true);

    const controller = sendMessageStream(activeSession, content, provider, {
      onToken: (token) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = {
              ...last,
              content: last.content + token,
            };
          }
          return updated;
        });
      },
      onSources: (sources) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = { ...last, sources };
          }
          return updated;
        });
      },
      onDone: () => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = { ...last, isStreaming: false };
          }
          return updated;
        });
        setIsGenerating(false);
      },
      onError: (error) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = {
              ...last,
              content: `Sorry, something went wrong: ${error}`,
              isStreaming: false,
            };
          }
          return updated;
        });
        setIsGenerating(false);
      },
    });

    abortRef.current = controller;
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-500 border-t-transparent" />
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="flex h-screen">
      <Sidebar
        currentSessionId={sessionId || undefined}
        onSessionSelect={loadSession}
        onNewChat={handleNewChat}
      />

      <div className="flex flex-1 flex-col">
        {messages.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center px-4">
            <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-100">
              <svg
                className="h-8 w-8 text-brand-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"
                />
              </svg>
            </div>
            <h2 className="mb-2 text-xl font-semibold text-gray-900">
              Ask your business coach
            </h2>
            <p className="mb-8 max-w-md text-center text-sm text-gray-500">
              Ask me anything about career advancement, leadership
              development, executive presence, or promotion strategies
              based on Tiffany Cheng&apos;s coaching videos.
            </p>
            <div className="grid gap-2 sm:grid-cols-2">
              {[
                "How do I get promoted to VP or senior director?",
                "What mindset shift is needed for executive roles?",
                "How do I develop executive presence?",
                "What separates team leaders from business leaders?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  className="rounded-lg border border-gray-200 px-4 py-3 text-left text-sm text-gray-600 transition hover:border-brand-300 hover:bg-brand-50"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto scrollbar-thin">
            <div className="mx-auto max-w-3xl">
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  role={msg.role}
                  content={msg.content}
                  sources={msg.sources}
                  isStreaming={msg.isStreaming}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        <ChatInput
          onSend={handleSend}
          disabled={isGenerating}
          provider={provider}
          onProviderChange={setProvider}
        />
      </div>
    </div>
  );
}
