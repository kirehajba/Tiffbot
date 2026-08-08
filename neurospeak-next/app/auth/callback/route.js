import { NextResponse } from "next/server";
import { supabaseFromCookies } from "../../../lib/supabaseServer";

export async function GET(req) {
  const url = new URL(req.url);
  const code = url.searchParams.get("code");
  if (code) {
    await supabaseFromCookies().auth.exchangeCodeForSession(code);
  }
  return NextResponse.redirect(new URL("/", url.origin));
}
