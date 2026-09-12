-- The corpus index. The text itself is not here.
--
-- Neon free is 0.5 GB in total and 5 GB of egress per month, and every Actions run that
-- reads from it spends both. So this table holds only what retrieval needs to decide
-- which units to fetch, and the unit text lives in the private HF dataset repo, addressed
-- by the shard column.
--
-- unit_id is a natural key derived from repo, kind and the unit's own identity, and it
-- deliberately does not include a commit sha. Including one would make every new HEAD
-- produce a fresh set of rows, and re-ingest would never converge. What moves when the
-- content moves is content_hash, and that is what the idempotence gate measures.
create table if not exists corpus_units (
    unit_id       text primary key,
    repo          text        not null,
    kind          text        not null,
    ref           text,
    path          text,
    symbol        text,
    start_line    integer,
    end_line      integer,
    content_hash  text        not null,
    char_count    integer     not null,
    shard         text        not null,
    metadata      jsonb       not null default '{}'::jsonb,
    first_seen_at timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);

create index if not exists corpus_units_repo_kind_idx on corpus_units (repo, kind);
create index if not exists corpus_units_content_hash_idx on corpus_units (content_hash);
create index if not exists corpus_units_path_idx on corpus_units (repo, path);

-- One row per ingest of one repo. This is the run log, and with no dataset viewer on a
-- private HF repo it is also the only durable record of what a mining run actually did.
create table if not exists ingest_runs (
    run_id      bigserial primary key,
    repo        text        not null,
    ref         text,
    started_at  timestamptz not null default now(),
    finished_at timestamptz,
    units_seen  integer     not null default 0,
    inserted    integer     not null default 0,
    updated     integer     not null default 0,
    unchanged   integer     not null default 0,
    shards      integer     not null default 0,
    error       text
);

create index if not exists ingest_runs_repo_started_idx on ingest_runs (repo, started_at desc);

create extension if not exists vector;

-- Phase 2 fills this. It is separate from corpus_units so that a change of embedding
-- model is a truncate rather than a schema migration on the table retrieval joins to.
--
-- 384 dimensions fits bge-small-en-v1.5 and all-MiniLM-L6-v2, both of which run free on
-- an Actions runner.
--
-- Settled 12 Sep 2026, with the corpus measured rather than estimated: this table holds
-- vectors only for repos the App is installed on, never the mining set. At 153,758 units,
-- heap plus key plus an HNSW index at m=16 is 521 MB at 384 dims and 993 MB at 768,
-- against 500 MB for the whole project, and reading them back per Actions run would spend
-- the 5 GB monthly egress cap in about 37 runs. The mining set's vectors live in the
-- private dataset repo and are searched exactly, in memory, on the runner. One installed
-- repo is about 5,700 units, 19 MB at 384 dims, so ten installs fit easily. If the
-- long-context model wins the Phase 2 table, 0003 moves this column to vector(768).
create table if not exists corpus_embeddings (
    unit_id   text primary key references corpus_units (unit_id) on delete cascade,
    model     text        not null,
    embedding vector(384) not null,
    created_at timestamptz not null default now()
);
