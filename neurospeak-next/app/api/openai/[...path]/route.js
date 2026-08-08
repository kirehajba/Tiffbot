import { currentUser, supabaseAdmin } from "../../../../lib/supabaseServer";

// The only OpenAI endpoints the app uses. Everything else is refused.
const ALLOWED_PATHS = new Set(["chat/completions", "audio/transcriptions", "audio/speech"]);
const ALLOWED_MODELS = new Set(["gpt-4o", "gpt-4o-mini", "gpt-4o-transcribe", "gpt-4o-mini-tts"]);

export async function POST(req, { params }) {
  const path = (params.path || []).join("/");
  if (!ALLOWED_PATHS.has(path)) {
    return Response.json({ error: "Endpoint not allowed" }, { status: 403 });
  }

  const user = await currentUser();
  if (!user) return Response.json({ error: "Sign in required" }, { status: 401 });

  // Metering: hard daily cap per user, atomic increment.
  const cap = parseInt(process.env.DAILY_REQUEST_CAP || "200", 10);
  const { data: count, error: meterError } = await supabaseAdmin().rpc("bump_usage", { uid: user.id });
  if (meterError) return Response.json({ error: "Metering unavailable" }, { status: 503 });
  if (count > cap) {
    return Response.json(
      { error: { message: "Daily practice limit reached — resets at midnight UTC." } },
      { status: 429 },
    );
  }

  const headers = { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` };
  let body;
  const contentType = req.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await req.json();
    if (payload.model && !ALLOWED_MODELS.has(payload.model)) {
      return Response.json({ error: "Model not allowed" }, { status: 403 });
    }
    if (typeof payload.max_tokens === "number") payload.max_tokens = Math.min(payload.max_tokens, 2000);
    payload.stream = false;
    body = JSON.stringify(payload);
    headers["Content-Type"] = "application/json";
  } else {
    // Multipart audio upload — forward verbatim, boundary intact.
    body = await req.arrayBuffer();
    headers["Content-Type"] = contentType;
  }

  const upstream = await fetch(`https://api.openai.com/v1/${path}`, { method: "POST", headers, body });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") || "application/octet-stream" },
  });
}
