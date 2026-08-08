-- NeuroSpeak production schema. Run in the Supabase SQL editor.

-- Per-user app data: drill history, answer bank, meeting circuit.
create table if not exists user_data (
  user_id uuid primary key references auth.users (id) on delete cascade,
  history jsonb not null default '[]',
  answers jsonb not null default '[]',
  meeting jsonb,
  updated_at timestamptz not null default now()
);

alter table user_data enable row level security;

create policy "own data read" on user_data
  for select using (auth.uid() = user_id);
create policy "own data write" on user_data
  for insert with check (auth.uid() = user_id);
create policy "own data update" on user_data
  for update using (auth.uid() = user_id);

-- Usage metering: one row per user per day, bumped by the API proxy.
-- Written only by the service role (no RLS policies for users needed).
create table if not exists usage_daily (
  user_id uuid not null references auth.users (id) on delete cascade,
  day date not null default current_date,
  requests int not null default 0,
  primary key (user_id, day)
);

alter table usage_daily enable row level security;

create policy "own usage read" on usage_daily
  for select using (auth.uid() = user_id);

-- Atomic increment used by the proxy.
create or replace function bump_usage(uid uuid)
returns int
language plpgsql
security definer
as $$
declare current int;
begin
  insert into usage_daily (user_id, day, requests)
  values (uid, current_date, 1)
  on conflict (user_id, day)
  do update set requests = usage_daily.requests + 1
  returning requests into current;
  return current;
end;
$$;
