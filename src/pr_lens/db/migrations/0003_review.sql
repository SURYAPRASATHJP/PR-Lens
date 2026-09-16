-- One row per pull request reviewed, live or replayed, and one per draft it produced.
--
-- The run row is taken BEFORE any tokens are spent, the same rule as the deliveries guard:
-- two replay jobs racing on one pull request, or a replay batch re-run after a failure,
-- find the row already there and draft nothing. The partial unique indexes below are what
-- make that true, not the application.
--
-- Rows stay small. Neon free is 0.5 GB in total, so no diff, prompt or retrieved text is
-- stored: a run holds its counts, its timings and the model's short plan; a draft holds
-- the comment, which is under a hundred words by instruction.
create table if not exists review_runs (
    run_id            bigserial primary key,
    mode              text        not null check (mode in ('live', 'replay')),
    repo              text        not null,
    pr_number         integer     not null,
    head_sha          text        not null,
    -- Live runs answer one delivery. Replay runs belong to a named batch.
    delivery_id       text        references deliveries (delivery_id),
    batch             text,
    -- Where retrieval read from when it is not the reviewed repo: a seeded testbed PR
    -- retrieves from the repository it was copied out of.
    source_repo       text,
    source_pr         integer,
    finished_at       timestamptz,
    no_comment        text,
    detail            text        not null default '',
    plan              text        not null default '',
    hunks_shown       integer     not null default 0,
    hunks_dropped     integer     not null default 0,
    files_skipped     integer     not null default 0,
    past_shown        integer     not null default 0,
    prompt_tokens     integer     not null default 0,
    completion_tokens integer     not null default 0,
    provider          text,
    model             text,
    timings           jsonb       not null default '{}'::jsonb,
    -- What the sandbox could say: whether it ran, why not when it did not, and how many
    -- tests passed at the base and failed at the head. Empty when it was not asked.
    verification      jsonb       not null default '{}'::jsonb,
    -- Replay only: the human review comments left on this pull request, shown beside the
    -- drafts in the keep-or-kill file. The answer key, so it is stored here and never
    -- passed to the pipeline, which has no parameter that could carry it.
    reference         jsonb       not null default '[]'::jsonb,
    started_at        timestamptz not null default now(),
    check ((mode = 'live') = (delivery_id is not null)),
    check ((mode = 'replay') = (batch is not null))
);

create unique index if not exists review_runs_one_replay_per_batch
    on review_runs (batch, repo, pr_number) where mode = 'replay';

create unique index if not exists review_runs_one_run_per_delivery
    on review_runs (delivery_id) where delivery_id is not null;

create index if not exists review_runs_repo_pr_idx on review_runs (repo, pr_number);

create table if not exists drafts (
    draft_id       bigserial primary key,
    run_id         bigint      not null references review_runs (run_id) on delete cascade,
    position       smallint    not null,
    path           text        not null,
    line           integer     not null,
    body           text        not null,
    evidence       text        not null,
    critique       text        not null,
    specific       boolean     not null,
    non_obvious    boolean     not null,
    grounded       boolean     not null,
    fate           text        not null,
    filter_reason  text        not null default '',
    -- The user's keep-or-kill, filled by `verdicts import`. Null means not yet read.
    verdict        text        check (verdict in ('keep', 'kill')),
    verdict_reason text,
    verdict_at     timestamptz,
    unique (run_id, position)
);
