-- What the drafting loop looked up before it answered.
--
-- "Drafts citing a tool result" is the headline number for Phase 4b, and the baseline it
-- has to beat is batch 2026-09-16-a: one keep, eight kills, every kill a claim about code
-- the model could not see. A number nobody can query is not a measurement, so the turns
-- the loop took and the calls it made are stored beside the run that made them.
--
-- Nullable and defaulted, because every run before this one took no turns and looked
-- nothing up, and recording that as zero would be a lie about what was measured.
alter table review_runs add column if not exists tool_turns integer;
alter table review_runs add column if not exists tools_used text;
