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
-- an Actions runner. Watch the row budget: 384 floats is about 1.5 kB, so 200k chunks is
-- 300 MB of a 500 MB database before the index. If it gets close, the vectors move to a
-- FAISS index in the dataset repo and this table keeps only what is queried live.
create table if not exists corpus_embeddings (
    unit_id   text primary key references corpus_units (unit_id) on delete cascade,
    model     text        not null,
    embedding vector(384) not null,
    created_at timestamptz not null default now()
);
