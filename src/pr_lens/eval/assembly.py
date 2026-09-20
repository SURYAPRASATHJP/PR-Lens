"""Did the hunk a human commented on survive into the prompt?

Retrieval recall is a ceiling the assembler can silently destroy. `review/context.py`
skips files, ranks hunks and then fits them greedily to a token budget, so a gold hunk can
be retrieved and then dropped before the model ever sees it. When that happens the review
comes back quiet, retrieval recall still reads 0.475@10, and nothing says which stage lost
it. This is the Phase 3 base-commit lesson again: without the comparison you blame the
wrong stage.

So every outcome is named, because the fix differs by outcome:

    SHOWN         the gold line was in a hunk that reached the prompt. The ceiling held.
    DROPPED       the hunk existed and ranked too low for the budget. Raise the budget,
                  change the ranking, or show a cheaper rendering.
    UNCONSIDERED  the hunk existed and ranked past MAX_CANDIDATE_HUNKS, so it was never
                  costed against the budget at all. Raise the ceiling, or rank better.
                  Batch 2026-09-18-b lost 263 hunks this way against 16 to the budget, so
                  folding the two together would point at the wrong fix.
    SKIPPED       the file was excluded before ranking, as a lockfile, vendored or
                  generated. Change skip_reason, not the budget.
    NOT_IN_DIFF   no hunk of this pull request contains that line. Not an assembly loss:
                  the human commented somewhere the diff never showed, so the ceiling was
                  never reachable and this row is excluded from the rate.

Nothing here is judged. Each query has one known gold location, the same ground truth the
recall table uses, so every number is arithmetic.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pr_lens.ingest.diff import Hunk, parse_diff_hunk
from pr_lens.review.context import Fitted, Skipped, commentable


class Outcome(StrEnum):
    SHOWN = "shown"
    DROPPED = "dropped"
    UNCONSIDERED = "unconsidered"
    SKIPPED = "skipped"
    NOT_IN_DIFF = "not_in_diff"


@dataclass(frozen=True, slots=True)
class Gold:
    """Where a human left a line comment: the ground truth for one query."""

    path: str
    line: int


def _contains(hunk: Hunk, gold: Gold) -> bool:
    """Whether a comment at gold could have been anchored inside this hunk.

    Membership is the commentable set, not the hunk's line span. A span includes removed
    lines, which carry no new-file number, so span arithmetic would credit the assembler
    with showing a line GitHub would refuse a comment on.
    """
    return hunk.path == gold.path and gold.line in commentable(hunk)


def classify(
    gold: Gold,
    fitted: Fitted,
    skipped: Sequence[Skipped] = (),
    unconsidered: Sequence[Hunk] = (),
) -> Outcome:
    """Where one gold comment ended up. Checked in the order the pipeline decides."""
    if any(_contains(hunk, gold) for hunk, _ in fitted.shown):
        return Outcome.SHOWN
    if any(_contains(hunk, gold) for hunk in fitted.dropped):
        return Outcome.DROPPED
    if any(_contains(hunk, gold) for hunk in unconsidered):
        return Outcome.UNCONSIDERED
    if any(entry.path == gold.path for entry in skipped):
        return Outcome.SKIPPED
    return Outcome.NOT_IN_DIFF


def rank_of(gold: Gold, ranked: Sequence[Hunk]) -> int | None:
    """Where the gold hunk sits in the ranking, counting from one, or None if no hunk holds
    it. This is the number MAX_CANDIDATE_HUNKS is a cut through, so the distribution of it
    is what says whether raising the ceiling would reach anything."""
    for position, hunk in enumerate(ranked, start=1):
        if _contains(hunk, gold):
            return position
    return None


def commented_line(diff_hunk: str, path: str) -> int | None:
    """The new-file line a review comment was left on.

    GitHub truncates a comment's diff_hunk so that it ends at the commented line, so the
    last line that exists in the new file is the one the human pointed at. Derived rather
    than stored because the frozen query set predates this measurement and re-mining it to
    add one field would change its digest and invalidate the recall table.
    """
    hunk = parse_diff_hunk(diff_hunk, path)
    if hunk is None:
        return None
    lines = hunk.new_file_lines()
    return lines[-1][0] if lines else None


@dataclass(frozen=True, slots=True)
class Assembly:
    """One row of the assembly table."""

    shown: int
    dropped: int
    skipped: int
    not_in_diff: int
    unconsidered: int = 0

    @property
    def reachable(self) -> int:
        """Queries whose gold was in the diff at all, so the assembler could have shown it."""
        return self.shown + self.dropped + self.skipped + self.unconsidered

    @property
    def retention(self) -> float:
        """Of the reachable queries, the fraction that reached the prompt.

        This is the number the recall table's 0.475@10 is an upper bound on. Reported as
        0.0 when nothing was reachable, which is not a score of zero: read `reachable`
        beside it, the way the recall table's counts are read beside its rates.
        """
        return self.shown / self.reachable if self.reachable else 0.0

    @property
    def total(self) -> int:
        return self.reachable + self.not_in_diff

    def row(self, label: str = "") -> str:
        cells = [
            f"{self.retention:.3f}",
            self.shown,
            self.dropped,
            self.unconsidered,
            self.skipped,
            self.not_in_diff,
            self.total,
        ]
        body = " | ".join(str(cell) for cell in cells)
        return f"| {label} | {body} |" if label else f"| {body} |"


COLUMNS = ("retention", "shown", "dropped", "unconsidered", "skipped", "not in diff", "queries")

HEADER = "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS)


def header(label: str = "") -> str:
    """The table header, optionally with a leading label column for a per-repo breakdown."""
    if not label:
        return HEADER
    return "| " + " | ".join((label, *COLUMNS)) + " |\n|" + "---|" * (len(COLUMNS) + 1)


def summarise(outcomes: Iterable[Outcome]) -> Assembly:
    counts: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
    for outcome in outcomes:
        counts[outcome] += 1
    return Assembly(
        shown=counts[Outcome.SHOWN],
        dropped=counts[Outcome.DROPPED],
        skipped=counts[Outcome.SKIPPED],
        not_in_diff=counts[Outcome.NOT_IN_DIFF],
        unconsidered=counts[Outcome.UNCONSIDERED],
    )
