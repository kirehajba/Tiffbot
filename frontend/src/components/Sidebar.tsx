"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  Video,
  Settings,
  Plus,
  Trash2,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { useAuth } from "@/lib/auth-context";
import {
  listSessions,
  createSession,
  deleteSession,
  type SessionInfo,
} from "@/lib/api";

interface SidebarProps {
  currentSessionId?: string;
  onSessionSelect: (id: string) => void;
  onNewChat: () => void;
}

export default function Sidebar({
  currentSessionId,
  onSessionSelect,
  onNewChat,
}: SidebarProps) {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, logout } = useAuth();
  const pathname = usePathname();

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const data = await listSessions();
      setSessions(data);
    } catch {
      /* ignore */
    }
  };

  const handleNewChat = async () => {
    onNewChat();
    await loadSessions();
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (currentSessionId === id) {
        onNewChat();
      }
    } catch {
      /* ignore */
    }
  };

  const navItems = [
    { href: "/chat", label: "Chat", icon: MessageSquare },
    { href: "/videos", label: "Videos", icon: Video },
    ...(user?.is_admin
      ? [{ href: "/admin", label: "Admin", icon: Settings }]
      : []),
  ];

  const sidebar = (
    <div className="flex h-full flex-col bg-gray-50 border-r border-gray-200">
      <div className="flex items-center justify-between p-4">
        <Link href="/chat" className="text-lg font-bold text-brand-600">
          TiiffBot
        </Link>
        <button
          className="lg:hidden rounded p-1 hover:bg-gray-200"
          onClick={() => setMobileOpen(false)}
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <nav className="px-3 space-y-1">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition",
              pathname === item.href
                ? "bg-brand-50 text-brand-700"
                : "text-gray-600 hover:bg-gray-100"
            )}
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </Link>
        ))}
      </nav>

      {pathname === "/chat" && (
        <>
          <div className="mt-4 px-3">
            <button
              onClick={handleNewChat}
              className="flex w-full items-center gap-2 rounded-lg border border-dashed border-gray-300 px-3 py-2 text-sm text-gray-600 transition hover:border-brand-400 hover:text-brand-600"
            >
              <Plus className="h-4 w-4" />
              New Chat
            </button>
          </div>

          <div className="mt-2 flex-1 overflow-y-auto px-3 scrollbar-thin">
            <div className="space-y-0.5 py-2">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => onSessionSelect(session.id)}
                  className={cn(
                    "group flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm transition",
                    currentSessionId === session.id
                      ? "bg-brand-50 text-brand-700"
                      : "text-gray-600 hover:bg-gray-100"
                  )}
                >
                  <span className="truncate">{session.title}</span>
                  <button
                    onClick={(e) => handleDelete(session.id, e)}
                    className="hidden shrink-0 rounded p-1 text-gray-400 hover:bg-gray-200 hover:text-red-500 group-hover:block"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      <div className="mt-auto border-t border-gray-200 p-3">
        <div className="flex items-center justify-between rounded-lg px-3 py-2">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-gray-900">
              {user?.name || user?.email}
            </p>
            {user?.name && (
              <p className="truncate text-xs text-gray-500">{user.email}</p>
            )}
          </div>
          <button
            onClick={logout}
            className="shrink-0 rounded p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
            title="Sign out"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile trigger */}
      <button
        className="fixed left-4 top-4 z-50 rounded-lg bg-white p-2 shadow-md lg:hidden"
        onClick={() => setMobileOpen(true)}
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/30 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile sidebar */}
      <div
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-72 transform transition-transform lg:hidden",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {sidebar}
      </div>

      {/* Desktop sidebar */}
      <div className="hidden w-72 shrink-0 lg:block">{sidebar}</div>
    </>
  );
}
