"use client";

import { useState } from "react";
import { createBrowserClient } from "@supabase/ssr";

function supabase() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  );
}

export default function Login() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState("");

  async function sendMagicLink(e) {
    e.preventDefault();
    setStatus("sending");
    const { error } = await supabase().auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${location.origin}/auth/callback` },
    });
    setStatus(error ? `Error: ${error.message}` : "sent");
  }

  async function googleSignIn() {
    await supabase().auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${location.origin}/auth/callback` },
    });
  }

  return (
    <main className="login-wrap">
      <h1>🧠 NeuroSpeak</h1>
      <p>Five minutes a day rewires how you speak. Sign in to start training.</p>
      <form onSubmit={sendMagicLink}>
        <input
          type="email"
          required
          placeholder="you@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <button type="submit" disabled={status === "sending"}>
          {status === "sent" ? "Check your inbox ✉️" : "Email me a sign-in link"}
        </button>
      </form>
      <button className="google" onClick={googleSignIn}>Continue with Google</button>
      {status.startsWith("Error") && <p className="err">{status}</p>}
    </main>
  );
}
