# NeuroSpeak — Production (Phase 1)

The productized version of the NeuroSpeak communication coach: same proven app, now with **server-held OpenAI key, user accounts, cloud-synced progress, and usage metering**. Users sign in and train — no API key required from them.

## Architecture

- **Next.js 14 (App Router)** on Vercel — one deployable unit
- **Client**: the battle-tested vanilla JS app (`public/app.js` + `app/shell.js`), served behind auth. No React rewrite — the working UI was kept as-is, with three surgical changes:
  1. every `api.openai.com` call now goes to `/api/openai/*` (authenticated proxy)
  2. no API-key dialog — the key lives in a server env var
  3. every local save also syncs to the cloud (debounced)
- **`/api/openai/[...path]`** — authenticated pass-through to OpenAI with an **endpoint allowlist** (chat, transcription, TTS), **model allowlist**, token clamp, and **per-user daily request cap** (atomic counter in Postgres)
- **Supabase** — auth (magic link + Google) and Postgres for `user_data` (history, answer bank, meeting circuit; RLS-protected) and `usage_daily` metering

## Setup

1. **Supabase**: create a free project → run `sql/schema.sql` in the SQL editor → enable Google provider (optional) under Auth → copy URL + keys.
2. **Env**: `cp .env.example .env` and fill in `OPENAI_API_KEY` + Supabase values.
3. **Run**: `npm install && npm run dev` → http://localhost:3000 (redirects to `/login`).
4. **Deploy**: push to GitHub → import in Vercel → add the same env vars → set the Vercel URL in Supabase Auth → URL configuration (redirect URLs).

## What Phase 2 adds (next)

- Stripe subscription (free tier: daily drills + 1 mock interview/mo; Pro: unlimited)
- Tier-aware caps in the proxy (replace the flat `DAILY_REQUEST_CAP`)
- Landing page + onboarding flow
- Sentry, prompt test-set, GDPR pages

## Regenerating the client from the single-file app

`public/app.js`, `app/shell.js`, and `app/globals.css` are generated from the
canonical single-file app (`docs/index.html` in the neuro-articulation-coach
repo) by the port script in the project history — keep iterating on the
single-file version, then re-run the transform to update production.
