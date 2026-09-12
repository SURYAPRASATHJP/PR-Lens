# The Phase 2 query set

Written down before any number was measured on it, which is the only point at which
writing it down is worth anything. The code that builds it is `src/pr_lens/eval/pairs.py`
and `src/pr_lens/jobs/pairs.py`; this page says what that code does and where it is weak.

## What a test case is

A query is a diff hunk a human reviewed. Its one gold document is the review comment they
left on it. That is the production direction: at review time PR-Lens holds a hunk and wants
to know what a reviewer would say about it. Drafting and distillation later need the same
direction, and the cross-encoder in the second stage is not symmetric, so the reverse trip,
comment to code, would not be a safe stand-in.

Recall is scored against the known gold id. Nothing is judged and no model grades anything.

## Where the pairs come from

The raw review-comment payloads, fetched through the Phase 1 GitHub client with the same
listing parameters Phase 1 used, so the Phase 1 cache answers most of it. Not from the
corpus units, which lack `in_reply_to_id`, the PR author and the merge state. Adding those
to unit metadata would change the content hash of every review comment unit and re-upload
every review comment shard.

A comment becomes a pair only if its gold unit is in the corpus snapshot. A comment newer
than the snapshot could only ever score zero.

## The rules, in order

Each comment is dropped by the first rule it fails, and the count per rule is recorded
in the frozen manifest.

| rule | why |
|---|---|
| no path or empty `diff_hunk` | nothing to query with |
| a reply (`in_reply_to_id` set) | thread roots only; a reply answers the thread, not the hunk |
| a bot | `[bot]` logins, `type: Bot`, a named list, and accounts that post templated machine reviews as a `User`, listed with where they were found |
| emoji only, an acknowledgement, or under 40 characters | nits and rubber stamps carry no review content |
| an empty before-state | an all-addition hunk has no pre-image, so the before-state query would be empty |
| the body quotes its own hunk | the body-only gold would contain its query |
| the pull request did not merge | a rejected change is a rejected idea |
| the comment author opened the pull request | self-review is not review |

Then two caps. At most three pairs per pull request, the earliest by comment id, because one
heavily reviewed pull request is one conversation, not many independent tests. And every
repo is capped at the median repo's surviving count, sampled by a hash of the comment id,
because review velocity across the tune repos spans two orders of magnitude and uncapped the
set measures two repos.

## Two queries per pair

`before` is the hunk's pre-image: context and removed lines, markers stripped, additions
dropped, the `@@` heading kept. `reviewed` is the `diff_hunk` exactly as the reviewer saw it.
The table reports both, on the same pairs.

They differ because of what a review comment's `diff_hunk` is. It is frozen at the commit the
comment was made on and ends at the commented line, which is almost always an added line.
The `+` lines are the code under review, not a later fix. Dropping them usually drops the
subject of the comment: a comment on a new decorator keeps only the unchanged lines above it.
Checked on live payloads on 12 Sep 2026. Which query is the headline is a decision the
numbers should inform rather than precede.

## Leakage, and how it is kept out

A review comment unit's indexed text is `path:line`, then the `diff_hunk`, then the body. It
contains the hunk the query is built from, so a query against that index finds the gold by
string match. The table therefore reports every first-stage row twice, against the corpus
as indexed and against the same corpus with review comments re-serialised body only. The gap
is the leakage. The body-only rows are the honest numbers.

Enforced, not promised. Before any body-only number is computed on the real query set, the
recall job asserts that no gold document contains either of its queries, and a test proves
the check fires on the with-hunk text. Other units from the same pull request stay in the
index as distractors.

## The repo split

18 tune repos that Phase 2 may see, 9 holdout repos it never embeds, queries or reads, in
`src/pr_lens/eval/split.py`. Assigned by ranking the 27 mining repos on review-comment
velocity and taking every third for the holdout. Split by repo, never by pull request.
Every Phase 2 entry point passes its repo list through `require_tune`, and a test runs the
recall job with a holdout repo slipped into its list and asserts that it refuses.

## Frozen

The pair set is written once to the private dataset repo as `eval/pairs/v1.jsonl.gz`, with a
manifest carrying its sha256, the cap and the per-rule counts. A second run with the same
version reads it back and changes nothing. Every table states the digest of the set it was
measured on.

## Weaknesses, stated plainly

- **Unfiltered for whether the comment was acted on.** Phase 5's stricter ground truth keeps
  only comments whose hunk the author then changed. That predicate is not validated yet, so
  this set includes comments that were ignored, wrong, or matters of taste. The table can
  gain an acted-on column later.
- **One gold per query.** A hunk can deserve several comments, and a different but equally
  good comment somewhere in the corpus scores as a miss. Recall here is a lower bound.
- **A window, not a history.** The corpus holds the most recent thousand review comments per
  repo, so older reviewers and older conventions are absent.
- **Suggestion blocks.** A comment carrying a GitHub suggestion block contains code that
  overlaps the hunk lexically. They stay in, and the manifest counts them.
- **The bot list is empirical.** Automated reviewers posting as ordinary users are only caught
  once someone sees one. New ones join the list and the fixture set the day they are found.
- **Reviewed code, not unreviewed.** These are merged, reviewed pull requests. Production
  serves code nobody has reviewed yet, which is the named threat to validity for the whole
  project, and it applies here too.
