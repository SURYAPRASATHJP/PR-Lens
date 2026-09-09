-- One row per pull request event the Actions job was asked to review.
--
-- The receiver does not write here. It verifies the signature and dispatches, and this
-- row is the Actions job's first act. That means the row's existence is also the
-- redelivery guard: GitHub redelivers, and the second run finds the id already present
-- and stops before anything comments twice.
--
-- Rows stay small. Neon free is 0.5 GB in total and this table grows with traffic, so no
-- diff or file text belongs in it.
create table if not exists deliveries (
    delivery_id     text primary key,
    event           text        not null,
    action          text,
    repo_full_name  text,
    pr_number       integer,
    head_sha        text,
    installation_id bigint,
    received_at     timestamptz not null default now()
);

-- "what have we already seen for this pull request" is asked on every review run.
create index if not exists deliveries_repo_pr_idx
    on deliveries (repo_full_name, pr_number);

create index if not exists deliveries_received_at_idx
    on deliveries (received_at desc);
