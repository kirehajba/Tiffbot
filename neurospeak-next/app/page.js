"use client";

import Script from "next/script";
import { SHELL } from "./shell";

// The app shell is the proven vanilla UI; /app.js drives it.
// Auth is enforced in two places: app.js redirects to /login on 401,
// and every /api/* route rejects unauthenticated calls server-side.
export default function Home() {
  return (
    <>
      <div dangerouslySetInnerHTML={{ __html: SHELL }} />
      <Script src="/app.js" strategy="afterInteractive" />
    </>
  );
}
