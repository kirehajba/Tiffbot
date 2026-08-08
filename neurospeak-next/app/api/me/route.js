import { currentUser } from "../../../lib/supabaseServer";

export async function GET() {
  const user = await currentUser();
  if (!user) return Response.json({ error: "unauthenticated" }, { status: 401 });
  return Response.json({ id: user.id, email: user.email });
}
