const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function getAuthHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
      ...options.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }

  return res.json();
}

// --- Auth ---

export async function signup(email: string, password: string, name?: string) {
  return request<{
    access_token: string;
    user: UserInfo;
  }>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
}

export async function login(email: string, password: string) {
  return request<{
    access_token: string;
    user: UserInfo;
  }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function getMe() {
  return request<UserInfo>("/api/auth/me");
}

// --- Chat sessions ---

export async function createSession() {
  return request<SessionInfo>("/api/chat/sessions", { method: "POST" });
}

export async function listSessions() {
  return request<SessionInfo[]>("/api/chat/sessions");
}

export async function getSession(sessionId: string) {
  return request<SessionDetail>(`/api/chat/sessions/${sessionId}`);
}

export async function deleteSession(sessionId: string) {
  return request(`/api/chat/sessions/${sessionId}`, { method: "DELETE" });
}

// --- Chat streaming ---

export function sendMessageStream(
  sessionId: string,
  content: string,
  provider: string,
  callbacks: {
    onToken: (token: string) => void;
    onSources: (sources: VideoSource[]) => void;
    onDone: () => void;
    onError: (error: string) => void;
  }
): AbortController {
  const abortController = new AbortController();
  const token = localStorage.getItem("token");

  fetch(`${BACKEND_URL}/api/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ content, provider }),
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        callbacks.onError(body.detail || "Request failed");
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const data = line.slice(5).trim();
            switch (currentEvent) {
              case "token":
                callbacks.onToken(data);
                break;
              case "sources":
                try {
                  callbacks.onSources(JSON.parse(data));
                } catch {
                  /* ignore parse errors */
                }
                break;
              case "done":
                callbacks.onDone();
                break;
              case "error":
                callbacks.onError(data);
                break;
            }
          }
        }
      }

      callbacks.onDone();
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return abortController;
}

// --- Videos ---

export async function listVideos(search = "", skip = 0, limit = 20) {
  const params = new URLSearchParams({
    search,
    skip: String(skip),
    limit: String(limit),
  });
  return request<{ videos: VideoInfo[]; total: number }>(
    `/api/videos?${params}`
  );
}

export async function getIngestionStatus() {
  return request<IngestionStatus>("/api/videos/status");
}

export async function triggerIngestion() {
  return request<{ message: string; videos_found: number }>(
    "/api/videos/ingest",
    { method: "POST" }
  );
}

// --- Types ---

export interface UserInfo {
  id: string;
  email: string;
  name: string | null;
  is_admin: boolean;
}

export interface SessionInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface MessageInfo {
  id: string;
  role: string;
  content: string;
  sources: string | null;
  created_at: string;
}

export interface SessionDetail {
  id: string;
  title: string;
  messages: MessageInfo[];
  created_at: string;
  updated_at: string;
}

export interface VideoSource {
  video_id: string;
  youtube_id: string;
  title: string;
  thumbnail_url: string;
  timestamp_seconds: number;
  relevance_score: number;
}

export interface VideoInfo {
  id: string;
  youtube_id: string;
  title: string;
  description: string | null;
  thumbnail_url: string | null;
  channel_title: string | null;
  published_at: string | null;
  duration: string | null;
  transcript_status: string;
  chunk_count: number;
}

export interface IngestionStatus {
  total_videos: number;
  completed: number;
  failed: number;
  pending: number;
  total_chunks: number;
}
