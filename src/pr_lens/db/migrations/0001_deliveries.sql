-- Every webhook GitHub sends us, recorded before we decide whether to act on it.
-- Rows are small on purpose. Neon free is 0.5 GB total and this table grows with
-- traffic, so nothing here holds diff or file text.
create table if not exists deliveries (
    delivery_id     text primary key,
    event           text        not null,
    action          text,
    repo_full_name  text,
    pr_number       integer,
    head_sha        text,
    installation_id bigint,
    received_at     timestamptz not null default now(),
    dispatched_at   timestamptz,
    dispatch_status text
);

-- "what have we seen for this PR" is the question asked on every review run.
create index if not exists deliveries_repo_pr_idx
    on deliveries (repo_full_name, pr_number);

create index if not exists deliveries_received_at_idx
    on deliveries (received_at desc);
