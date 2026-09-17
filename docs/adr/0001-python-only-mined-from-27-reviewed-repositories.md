# ADR 0001: Python only, mined from 27 actively reviewed repositories

Date: 2026-09-09
Status: accepted. Language clause partly superseded 2026-09-14 (note below).
Measured set: `0001-mining-set.md`.

## Context

PR-Lens needs ground truth: merged pull requests where a human left a line comment and the
author then changed that hunk. That exists only in maintained multi-contributor
projects that review line by line.

Phase 7 serves the opposite: solo developers with no reviewer. Measured on reviewed code,
served on unreviewed code. That gap is the central threat to validity.

## Decision

Python only for v1: one test runner, one chunker, one metric set.

Twenty-seven repositories, selected on:

- review velocity: 100 recent review comments over the days they span, cutoff 0.1 a day
- permissive licence, active inside 45 days, domain spread
- not django, pandas or cpython, whose review is decade-old API stability work

The golden set splits by repository, never by pull request: comments in one repository share
reviewers, style and vocabulary.

## Rejected alternatives

**Adding TypeScript.** Two to three build days, every metric splits. Real cost: bootcamp
repositories skew to JS and TS.

**Small solo repositories.** Circular: no reviewers, nothing to mine.

**Issue comments.** Not line-anchored, so the changed-hunk signal cannot be computed.

## Consequences

The mining set is not the serving set, so these numbers overstate served performance. The
README says so.

Velocity spans 0.15 to 17.8 a day, so per-repository contribution must be capped.

Open: celery and pymc report NOASSERTION, neither confirmed, source mined anyway. Corpus is
private, nothing republished. Confirm or drop.

## Cost of reversing

Adding a language is additive. Changing the list is cheap until Phase 5 locks the golden
set; after that, re-mining invalidates every published number.

## Note, 2026-09-14

Phase 3 made the sandbox run Python and JavaScript or TypeScript, since an installed
repository is not in the mining set. Corpus and golden set stay Python only.
