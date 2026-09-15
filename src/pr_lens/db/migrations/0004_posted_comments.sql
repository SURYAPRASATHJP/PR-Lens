-- Every comment PR-Lens posts, or tried to, with its row taken BEFORE the API call.
--
-- Two review runs on one pull request are the normal case: a redelivery, two pushes, a
-- cancelled run that had already started posting. Cancelling a run cannot un-post a
-- comment, so the promise not to double-comment is kept here, as constraints, and the
-- application only has to take the row first and post second. A losing racer gets no row
-- and posts nothing.
--
--   the per-PR cap   the primary key is (repo, pr_number, slot) and slot can only be 1, 2
--                    or 3, so a fourth row on one pull request cannot exist. 3 is
--                    MAX_COMMENTS_PER_PR in review/pipeline.py; a test holds the two equal.
--   one per line     unique (repo, pr_number, path, line): a line PR-Lens has commented on
--                    is never commented on again, by any later run.
--
-- A row whose post failed keeps its slot. Retrying risks the duplicate this table exists
-- to prevent, and silence is a valid output where a duplicate is not.
create table if not exists posted_comments (
    repo              text        not null,
    pr_number         integer     not null,
    slot              smallint    not null check (slot between 1 and 3),
    path              text        not null,
    line              integer     not null,
    run_id            bigint      references review_runs (run_id) on delete set null,
    body_sha256       text        not null,
    status            text        not null default 'pending'
                                  check (status in ('pending', 'posted', 'failed')),
    github_comment_id bigint,
    error             text,
    created_at        timestamptz not null default now(),
    posted_at         timestamptz,
    primary key (repo, pr_number, slot),
    unique (repo, pr_number, path, line)
);
