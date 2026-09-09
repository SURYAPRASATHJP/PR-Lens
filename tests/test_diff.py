from pr_lens.ingest.diff import parse_diff, parse_diff_hunk, parse_patch

PATCH = """@@ -10,3 +10,4 @@ def parse(self):
     context
-    removed
+    added one
+    added two
"""


def test_a_hunk_header_is_read_exactly() -> None:
    hunk = parse_patch(PATCH, "pr_lens/parser.py")[0]
    assert (hunk.old_start, hunk.old_lines) == (10, 3)
    assert (hunk.new_start, hunk.new_lines) == (10, 4)
    assert hunk.section == "def parse(self):"
    assert hunk.path == "pr_lens/parser.py"


def test_added_lines_carry_their_line_number_in_the_new_file() -> None:
    hunk = parse_patch(PATCH, "x.py")[0]
    assert hunk.added() == [(11, "    added one"), (12, "    added two")]
    assert hunk.removed() == ["    removed"]


def test_an_omitted_count_means_one_line() -> None:
    hunk = parse_patch("@@ -5 +5 @@\n-a\n+b\n", "x.py")[0]
    assert (hunk.old_lines, hunk.new_lines) == (1, 1)
    assert hunk.added() == [(5, "b")]


def test_a_no_newline_marker_does_not_shift_the_line_numbers() -> None:
    patch = "@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n"
    assert parse_patch(patch, "x.py")[0].added() == [(2, "c")]


def test_a_multi_file_diff_attributes_each_hunk_to_its_own_file() -> None:
    diff = (
        "diff --git a/one.py b/one.py\n"
        "index abc..def 100644\n"
        "--- a/one.py\n"
        "+++ b/one.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/docs/two.md b/docs/two.md\n"
        "--- a/docs/two.md\n"
        "+++ b/docs/two.md\n"
        "@@ -3,1 +3,2 @@ Heading\n"
        " keep\n"
        "+extra\n"
    )
    hunks = parse_diff(diff)
    assert [h.path for h in hunks] == ["one.py", "docs/two.md"]
    assert hunks[1].added() == [(4, "extra")]


def test_a_new_file_has_no_old_side() -> None:
    diff = "diff --git a/n.py b/n.py\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n+one\n+two\n"
    hunk = parse_diff(diff)[0]
    assert hunk.path == "n.py"
    assert hunk.added() == [(1, "one"), (2, "two")]


def test_a_deleted_file_yields_no_attributable_hunk() -> None:
    diff = "diff --git a/g.py b/g.py\n--- a/g.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-one\n-two\n"
    assert parse_diff(diff) == []


def test_a_review_comments_truncated_hunk_is_kept_as_sent() -> None:
    # GitHub truncates diff_hunk, so its header routinely overstates the lines below it.
    # Correcting the header would put us out of step with GitHub's own line arithmetic.
    hunk = parse_diff_hunk("@@ -40,12 +40,14 @@ class Thing:\n     tail\n+    added\n", "t.py")
    assert hunk is not None
    assert hunk.new_lines == 14
    assert len(hunk.lines) == 2
    assert hunk.added() == [(41, "    added")]


def test_text_round_trips_a_hunk() -> None:
    hunk = parse_patch(PATCH, "x.py")[0]
    assert hunk.text.startswith("@@ -10,3 +10,4 @@ def parse(self):")
    assert "+    added two" in hunk.text


def test_empty_input_is_not_an_error() -> None:
    assert parse_diff("") == []
    assert parse_diff_hunk("", "x.py") is None
