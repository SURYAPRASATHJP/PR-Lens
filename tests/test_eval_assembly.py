"""The assembly metric: every outcome, and the ones that would quietly flatter it.

Each case differs from the shown baseline in one way, so a failure names the clause that
broke rather than the metric in general.
"""

from pr_lens.eval.assembly import (
    HEADER,
    Assembly,
    Gold,
    Outcome,
    classify,
    commented_line,
    rank_of,
    summarise,
)
from pr_lens.ingest.diff import Hunk
from pr_lens.review.context import Fitted, Skipped, commentable


def hunk(path: str = "a.py", new_start: int = 10, added: int = 3) -> Hunk:
    """A hunk whose new-file lines run from new_start, all added so all commentable."""
    return Hunk(
        path=path,
        old_start=new_start,
        old_lines=0,
        new_start=new_start,
        new_lines=added,
        section=None,
        lines=tuple(f"+line {new_start + offset}" for offset in range(added)),
    )


def removed_only(path: str = "a.py", old_start: int = 10) -> Hunk:
    """A hunk of removals. It spans lines but has no commentable new-file line."""
    return Hunk(
        path=path,
        old_start=old_start,
        old_lines=3,
        new_start=old_start,
        new_lines=0,
        section=None,
        lines=("-gone one", "-gone two", "-gone three"),
    )


def test_a_gold_line_inside_a_shown_hunk_is_shown() -> None:
    fitted = Fitted(shown=[(hunk(), "rendered")], dropped=[], tokens=100)
    assert classify(Gold("a.py", 11), fitted) is Outcome.SHOWN


def test_a_gold_line_in_a_dropped_hunk_is_dropped_not_shown() -> None:
    fitted = Fitted(shown=[], dropped=[hunk()], tokens=0)
    assert classify(Gold("a.py", 11), fitted) is Outcome.DROPPED


def test_a_skipped_file_is_skipped_not_not_in_diff() -> None:
    fitted = Fitted(shown=[], dropped=[], tokens=0)
    skipped = [Skipped(path="package-lock.json", reason="lockfile")]
    assert classify(Gold("package-lock.json", 4), fitted, skipped) is Outcome.SKIPPED


def test_a_line_no_hunk_touched_is_not_in_diff() -> None:
    fitted = Fitted(shown=[(hunk(), "rendered")], dropped=[], tokens=10)
    assert classify(Gold("a.py", 900), fitted) is Outcome.NOT_IN_DIFF


def test_the_same_line_number_in_another_file_does_not_count() -> None:
    """Path must match. Without this the metric credits a coincidence of line numbers."""
    fitted = Fitted(shown=[(hunk(path="a.py"), "rendered")], dropped=[], tokens=10)
    assert classify(Gold("b.py", 11), fitted) is Outcome.NOT_IN_DIFF


def test_a_removed_only_hunk_does_not_count_as_shown() -> None:
    """It spans the line but GitHub would refuse a comment there, so it is not coverage."""
    fitted = Fitted(shown=[(removed_only(), "rendered")], dropped=[], tokens=10)
    assert classify(Gold("a.py", 11), fitted) is Outcome.NOT_IN_DIFF


def test_a_line_just_past_the_commentable_set_is_not_covered() -> None:
    """The span reaches it, the commentable set does not, and the set is what GitHub takes.

    new_start 10 with new_lines 2 spans 10 to 12 but numbers only 10 and 11. Measuring by
    span would count line 12 as shown and quietly inflate retention.
    """
    h = hunk(new_start=10, added=2)
    assert sorted(commentable(h)) == [10, 11]
    assert h.new_start + h.new_lines == 12
    fitted = Fitted(shown=[(h, "rendered")], dropped=[], tokens=10)
    assert classify(Gold("a.py", 12), fitted) is Outcome.NOT_IN_DIFF


def test_shown_beats_dropped_when_one_hunk_is_in_both_lists() -> None:
    """Order of the checks, pinned. A hunk that reached the prompt is not an assembly loss."""
    h = hunk()
    fitted = Fitted(shown=[(h, "rendered")], dropped=[h], tokens=10)
    assert classify(Gold("a.py", 11), fitted) is Outcome.SHOWN


def test_shown_wins_when_a_later_hunk_of_the_same_file_was_dropped() -> None:
    fitted = Fitted(
        shown=[(hunk(new_start=10), "rendered")], dropped=[hunk(new_start=50)], tokens=10
    )
    assert classify(Gold("a.py", 11), fitted) is Outcome.SHOWN
    assert classify(Gold("a.py", 51), fitted) is Outcome.DROPPED


def test_retention_counts_only_reachable_queries() -> None:
    """not_in_diff is excluded: the assembler was never able to show it."""
    stats = summarise(
        [Outcome.SHOWN, Outcome.SHOWN, Outcome.DROPPED, Outcome.SKIPPED, Outcome.NOT_IN_DIFF]
    )
    assert stats.reachable == 4
    assert stats.total == 5
    assert stats.retention == 0.5


def test_retention_is_zero_and_not_an_error_when_nothing_was_reachable() -> None:
    stats = summarise([Outcome.NOT_IN_DIFF, Outcome.NOT_IN_DIFF])
    assert stats.reachable == 0
    assert stats.retention == 0.0
    assert stats.total == 2


def test_perfect_and_total_loss_read_as_they_should() -> None:
    assert summarise([Outcome.SHOWN, Outcome.SHOWN]).retention == 1.0
    assert summarise([Outcome.DROPPED, Outcome.SKIPPED]).retention == 0.0


def test_the_row_matches_the_header_width() -> None:
    row = Assembly(shown=2, dropped=1, skipped=1, not_in_diff=1).row()
    assert row.count("|") == HEADER.splitlines()[0].count("|")
    assert "0.500" in row


def test_a_gold_past_the_candidate_ceiling_is_unconsidered_not_not_in_diff() -> None:
    """The distinction the whole measurement turns on. Folded into NOT_IN_DIFF it reads as
    "the human commented outside the diff", which excludes it from the rate and says the
    assembler lost nothing. It is a loss, and the fix is the ceiling, not the budget."""
    fitted = Fitted(shown=[(hunk(new_start=10), "rendered")], dropped=[], tokens=10)
    beyond = [hunk(path="z.py", new_start=500)]
    assert classify(Gold("z.py", 501), fitted, (), beyond) is Outcome.UNCONSIDERED
    assert classify(Gold("z.py", 501), fitted) is Outcome.NOT_IN_DIFF


def test_unconsidered_counts_as_reachable_because_a_higher_ceiling_would_reach_it() -> None:
    stats = summarise([Outcome.SHOWN, Outcome.UNCONSIDERED])
    assert stats.reachable == 2
    assert stats.retention == 0.5


def test_shown_beats_unconsidered_when_the_same_file_appears_twice() -> None:
    fitted = Fitted(shown=[(hunk(new_start=10), "rendered")], dropped=[], tokens=10)
    beyond = [hunk(new_start=50)]
    assert classify(Gold("a.py", 11), fitted, (), beyond) is Outcome.SHOWN
    assert classify(Gold("a.py", 51), fitted, (), beyond) is Outcome.UNCONSIDERED


def test_rank_counts_from_one_and_is_none_when_no_hunk_holds_the_line() -> None:
    ranked = [hunk(path="a.py", new_start=10), hunk(path="b.py", new_start=20)]
    assert rank_of(Gold("a.py", 11), ranked) == 1
    assert rank_of(Gold("b.py", 21), ranked) == 2
    assert rank_of(Gold("c.py", 1), ranked) is None


def test_the_commented_line_is_the_last_line_of_the_diff_hunk() -> None:
    """GitHub truncates diff_hunk so it ends at the commented line, which is why the line
    can be recovered from the frozen query set without re-mining it."""
    diff_hunk = "@@ -1,3 +1,4 @@\n context\n-gone\n+added one\n+added two"
    assert commented_line(diff_hunk, "a.py") == 3


def test_a_removed_last_line_does_not_become_the_commented_line() -> None:
    """A removed line carries no new-file number, so it cannot be what GitHub anchored to."""
    diff_hunk = "@@ -1,3 +1,2 @@\n context\n+added\n-gone"
    assert commented_line(diff_hunk, "a.py") == 2


def test_an_unparseable_diff_hunk_has_no_gold_line() -> None:
    assert commented_line("not a diff at all", "a.py") is None
