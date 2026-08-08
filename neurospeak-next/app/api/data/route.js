import { currentUser, supabaseFromCookies } from "../../../lib/supabaseServer";

/** GET: the user's synced app data (history, answers, meeting). */
export async function GET() {
  const user = await currentUser();
  if (!user) return Response.json({ error: "unauthenticated" }, { status: 401 });
  const { data, error } = await supabaseFromCookies()
    .from("user_data").select("history, answers, meeting").eq("user_id", user.id).maybeSingle();
  if (error) return Response.json({ error: error.message }, { status: 500 });
  return Response.json(data || { history: [], answers: [], meeting: null });
}

/** POST: replace the user's synced snapshot (client is source of truth). */
export async function POST(req) {
  const user = await currentUser();
  if (!user) return Response.json({ error: "unauthenticated" }, { status: 401 });
  const body = await req.json();
  const row = {
    user_id: user.id,
    history: Array.isArray(body.history) ? body.history.slice(-2000) : [],
    answers: Array.isArray(body.answers) ? body.answers.slice(-500) : [],
    meeting: body.meeting ?? null,
    updated_at: new Date().toISOString(),
  };
  const { error } = await supabaseFromCookies().from("user_data").upsert(row);
  if (error) return Response.json({ error: error.message }, { status: 500 });
  return Response.json({ ok: true });
}
