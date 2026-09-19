-- Tell a hunk the budget could not fit from one the ranking never looked at.
--
-- hunks_dropped has been counting both since Phase 4, so batch 2026-09-18-b reads as an
-- average of 18.6 dropped hunks with one pull request at 192. That is MAX_CANDIDATE_HUNKS,
-- not the token budget, and the two want opposite fixes: one is a cheaper rendering, the
-- other is a higher ceiling. A metric that cannot tell them apart points at the wrong one.
alter table review_runs add column if not exists hunks_unconsidered integer;
