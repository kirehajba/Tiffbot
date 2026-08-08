import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import { createClient } from "@supabase/supabase-js";

/** Request-scoped client that reads the user's session from cookies. */
export function supabaseFromCookies() {
  const store = cookies();
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
    {
      cookies: {
        getAll: () => store.getAll(),
        setAll: (list) => {
          try { list.forEach(({ name, value, options }) => store.set(name, value, options)); }
          catch { /* Server Components can't set cookies; route handlers can. */ }
        },
      },
    },
  );
}

/** Service-role client for metering writes. Never expose to the browser. */
export function supabaseAdmin() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    { auth: { persistSession: false } },
  );
}

/** Resolve the authenticated user or null. */
export async function currentUser() {
  const { data, error } = await supabaseFromCookies().auth.getUser();
  return error ? null : data.user;
}
