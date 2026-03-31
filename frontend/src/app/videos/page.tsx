"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Search, ExternalLink, CheckCircle, AlertCircle, Clock } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import Sidebar from "@/components/Sidebar";
import { listVideos, type VideoInfo } from "@/lib/api";

export default function VideosPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [loadingVideos, setLoadingVideos] = useState(true);
  const limit = 12;

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [user, loading, router]);

  const fetchVideos = useCallback(async () => {
    setLoadingVideos(true);
    try {
      const data = await listVideos(search, page * limit, limit);
      setVideos(data.videos);
      setTotal(data.total);
    } catch {
      /* ignore */
    } finally {
      setLoadingVideos(false);
    }
  }, [search, page]);

  useEffect(() => {
    if (user) fetchVideos();
  }, [user, fetchVideos]);

  const statusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-emerald-500" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  if (loading || !user) return null;

  return (
    <div className="flex h-screen">
      <Sidebar
        onSessionSelect={() => router.push("/chat")}
        onNewChat={() => router.push("/chat")}
      />

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-6 py-8">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-gray-900">Video Library</h1>
            <p className="mt-1 text-sm text-gray-500">
              Browse all indexed coaching videos ({total} total)
            </p>
          </div>

          <div className="relative mb-6">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
              placeholder="Search videos..."
              className="w-full rounded-lg border border-gray-300 py-2.5 pl-10 pr-4 text-sm outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            />
          </div>

          {loadingVideos ? (
            <div className="flex justify-center py-12">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-brand-500 border-t-transparent" />
            </div>
          ) : videos.length === 0 ? (
            <div className="rounded-lg border border-dashed border-gray-300 py-12 text-center">
              <p className="text-gray-500">No videos found</p>
            </div>
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {videos.map((video) => (
                  <a
                    key={video.id}
                    href={`https://www.youtube.com/watch?v=${video.youtube_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group overflow-hidden rounded-xl border border-gray-200 bg-white transition hover:border-brand-300 hover:shadow-md"
                  >
                    {video.thumbnail_url && (
                      <div className="relative aspect-video overflow-hidden bg-gray-100">
                        <img
                          src={video.thumbnail_url}
                          alt={video.title}
                          className="h-full w-full object-cover transition group-hover:scale-105"
                        />
                        <div className="absolute right-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-xs text-white">
                          {video.chunk_count} chunks
                        </div>
                      </div>
                    )}
                    <div className="p-4">
                      <div className="mb-2 flex items-start justify-between gap-2">
                        <h3 className="line-clamp-2 text-sm font-medium text-gray-900 group-hover:text-brand-700">
                          {video.title}
                        </h3>
                        <ExternalLink className="mt-0.5 h-4 w-4 shrink-0 text-gray-400 group-hover:text-brand-500" />
                      </div>
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        {statusIcon(video.transcript_status)}
                        <span className="capitalize">
                          {video.transcript_status}
                        </span>
                        {video.published_at && (
                          <>
                            <span className="text-gray-300">|</span>
                            <span>{video.published_at}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </a>
                ))}
              </div>

              {total > limit && (
                <div className="mt-6 flex items-center justify-center gap-2">
                  <button
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="rounded-lg border border-gray-300 px-4 py-2 text-sm disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <span className="text-sm text-gray-500">
                    Page {page + 1} of {Math.ceil(total / limit)}
                  </span>
                  <button
                    onClick={() => setPage((p) => p + 1)}
                    disabled={(page + 1) * limit >= total}
                    className="rounded-lg border border-gray-300 px-4 py-2 text-sm disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
