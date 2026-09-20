"""The assembly runner's own logic: grouping the golds, and the table it renders."""

from collections import Counter

from pr_lens.eval.assembly import Assembly
from pr_lens.eval.pairs import ReviewPair
from pr_lens.jobs.assembly import golds_by_pull, render


def pair(
    repo: str = "o/r", number: int = 7, diff_hunk: str = "@@ -1,2 +1,2 @@\n a\n+b"
) -> ReviewPair:
    return ReviewPair(
        repo=repo,
        comment_id=1,
        pull_request=number,
        path="a.py",
        diff_hunk=diff_hunk,
        body="body long enough to survive the nit rules",
        author="someone",
        created_at="2026-01-01T00:00:00Z",
        html_url="https://example/1",
    )


def test_pairs_are_grouped_by_pull_request() -> None:
    grouped = golds_by_pull([pair(number=7), pair(number=7), pair(number=9)])
    assert sorted(grouped) == [("o/r", 7), ("o/r", 9)]
    assert len(grouped[("o/r", 7)]) == 2


def test_a_pair_whose_line_cannot_be_recovered_is_left_out_not_scored() -> None:
    """Counted as excluded input instead. Scored, it would read as NOT_IN_DIFF, which the
    table treats as "the ceiling was never reachable" and so would flatter retention."""
    grouped = golds_by_pull([pair(diff_hunk="not a diff")])
    assert grouped == {}


def test_the_table_reports_unconsidered_as_its_own_column() -> None:
    stats = Assembly(shown=3, dropped=1, skipped=1, not_in_diff=2, unconsidered=4)
    table = render(stats, {"o/r": stats}, Counter({1: 3, 12: 4}), unplaceable=2)
    assert "unconsidered" in table
    # Folding unconsidered into dropped would report 5 and point at the token budget.
    assert "| 3 | 1 | 4 | 1 | 2 |" in table
    assert "12 (past the cut)" in table
    assert "4 of 7 placed gold hunks ranked past 8" in table
    assert "2 pairs could not be placed" in table


def test_a_run_that_placed_nothing_renders_without_dividing_by_zero() -> None:
    stats = Assembly(shown=0, dropped=0, skipped=0, not_in_diff=0)
    table = render(stats, {}, Counter(), unplaceable=0)
    assert "0 of 0 placed gold hunks" in table
