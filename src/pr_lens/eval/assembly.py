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

from pr_lens.ingest.diff import Hunk
from pr_lens.review.context import Fitted, Skipped, commentable


class Outcome(StrEnum):
    SHOWN = "shown"
    DROPPED = "dropped"
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


def classify(gold: Gold, fitted: Fitted, skipped: Sequence[Skipped] = ()) -> Outcome:
    """Where one gold comment ended up. Checked in the order the pipeline decides."""
    if any(_contains(hunk, gold) for hunk, _ in fitted.shown):
        return Outcome.SHOWN
    if any(_contains(hunk, gold) for hunk in fitted.dropped):
        return Outcome.DROPPED
    if any(entry.path == gold.path for entry in skipped):
        return Outcome.SKIPPED
    return Outcome.NOT_IN_DIFF


@dataclass(frozen=True, slots=True)
class Assembly:
    """One row of the assembly table."""

    shown: int
    dropped: int
    skipped: int
    not_in_diff: int

    @property
    def reachable(self) -> int:
        """Queries whose gold was in the diff at all, so the assembler could have shown it."""
        return self.shown + self.dropped + self.skipped

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

    def row(self) -> str:
        return (
            f"| {self.retention:.3f} | {self.shown} | {self.dropped} | "
            f"{self.skipped} | {self.not_in_diff} | {self.total} |"
        )


HEADER = (
    "| retention | shown | dropped | skipped | not in diff | queries |\n|---|---|---|---|---|---|"
)


def summarise(outcomes: Iterable[Outcome]) -> Assembly:
    counts: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
    for outcome in outcomes:
        counts[outcome] += 1
    return Assembly(
        shown=counts[Outcome.SHOWN],
        dropped=counts[Outcome.DROPPED],
        skipped=counts[Outcome.SKIPPED],
        not_in_diff=counts[Outcome.NOT_IN_DIFF],
    )
