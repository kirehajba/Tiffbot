"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Database,
  Video,
  FileText,
  AlertTriangle,
  Clock,
  RefreshCw,
  CheckCircle,
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import Sidebar from "@/components/Sidebar";
import {
  getIngestionStatus,
  triggerIngestion,
  type IngestionStatus,
} from "@/lib/api";

export default function AdminPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [status, setStatus] = useState<IngestionStatus | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [message, setMessage] = useState("");
  const [loadingStatus, setLoadingStatus] = useState(true);

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
    if (!loading && user && !user.is_admin) router.replace("/chat");
  }, [user, loading, router]);

  useEffect(() => {
    if (user?.is_admin) loadStatus();
  }, [user]);

  const loadStatus = async () => {
    setLoadingStatus(true);
    try {
      const data = await getIngestionStatus();
      setStatus(data);
    } catch {
      /* ignore */
    } finally {
      setLoadingStatus(false);
    }
  };

  const handleIngest = async () => {
    setIngesting(true);
    setMessage("");
    try {
      const res = await triggerIngestion();
      setMessage(res.message);
      setTimeout(loadStatus, 3000);
    } catch (err: any) {
      setMessage(`Error: ${err.message}`);
    } finally {
      setIngesting(false);
    }
  };

  if (loading || !user?.is_admin) return null;

  const stats = [
    {
      label: "Total Videos",
      value: status?.total_videos ?? "-",
      icon: Video,
      color: "text-brand-600 bg-brand-50",
    },
    {
      label: "Completed",
      value: status?.completed ?? "-",
      icon: CheckCircle,
      color: "text-emerald-600 bg-emerald-50",
    },
    {
      label: "Failed",
      value: status?.failed ?? "-",
      icon: AlertTriangle,
      color: "text-red-600 bg-red-50",
    },
    {
      label: "Pending",
      value: status?.pending ?? "-",
      icon: Clock,
      color: "text-yellow-600 bg-yellow-50",
    },
    {
      label: "Total Chunks",
      value: status?.total_chunks ?? "-",
      icon: FileText,
      color: "text-purple-600 bg-purple-50",
    },
    {
      label: "Vector DB",
      value: status ? "Connected" : "-",
      icon: Database,
      color: "text-cyan-600 bg-cyan-50",
    },
  ];

  return (
    <div className="flex h-screen">
      <Sidebar
        onSessionSelect={() => router.push("/chat")}
        onNewChat={() => router.push("/chat")}
      />

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-6 py-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Admin Dashboard
              </h1>
              <p className="mt-1 text-sm text-gray-500">
                Manage video ingestion and monitor system status
              </p>
            </div>
            <button
              onClick={loadStatus}
              disabled={loadingStatus}
              className="rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-600 transition hover:bg-gray-50"
            >
              <RefreshCw
                className={`h-4 w-4 ${loadingStatus ? "animate-spin" : ""}`}
              />
            </button>
          </div>

          <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="rounded-xl border border-gray-200 bg-white p-5"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`flex h-10 w-10 items-center justify-center rounded-lg ${stat.color}`}
                  >
                    <stat.icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-gray-900">
                      {stat.value}
                    </p>
                    <p className="text-xs text-gray-500">{stat.label}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="rounded-xl border border-gray-200 bg-white p-6">
            <h2 className="mb-4 text-lg font-semibold text-gray-900">
              Video Ingestion
            </h2>
            <p className="mb-4 text-sm text-gray-600">
              Trigger a full ingestion of your YouTube channel. This will fetch
              all videos, extract transcripts, and index them for AI search.
              New videos will be added; existing ones are skipped.
            </p>

            {message && (
              <div className="mb-4 rounded-lg bg-blue-50 p-3 text-sm text-blue-700">
                {message}
              </div>
            )}

            <button
              onClick={handleIngest}
              disabled={ingesting}
              className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:opacity-50"
            >
              {ingesting ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Ingesting...
                </>
              ) : (
                <>
                  <Database className="h-4 w-4" />
                  Start Ingestion
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
