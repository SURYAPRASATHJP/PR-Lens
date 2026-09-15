from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient
from pr_lens.ingest.diff import parse_patch
from pr_lens.review.context import (
    MAX_HUNK_LINES,
    Block,
    commentable,
    fetch_changes,
    fetch_pull,
    fetch_windows,
    fit,
    rank,
    render,
    skip_reason,
)
from pr_lens.review.provider import estimate_tokens

API = "https://api.github.com"
PATCH = (
    "@@ -3,3 +3,4 @@ def run():\n     a = 1\n-    b = 2\n+    b = 3\n+    c = 4\n     return a\n"
)


def changed(filename: str, **fields: Any) -> dict[str, Any]:
    return {"filename": filename, "status": "modified", "changes": 2, "patch": PATCH, **fields}


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (changed("gone.py", status="removed"), "deleted"),
        (changed("new_name.py", status="renamed", changes=0), "renamed without changes"),
        (changed("web/package-lock.json"), "lockfile or generated"),
        (changed("uv.lock"), "lockfile or generated"),
        (changed("static/app.min.js"), "lockfile or generated"),
        (changed("tests/__snapshots__/view.snap"), "lockfile or generated"),
        (changed("src/vendor/lib.py"), "vendored"),
        (changed("logo.png", patch=None), "no patch: binary or too large"),
    ],
    ids=lambda value: value if isinstance(value, str) else value["filename"],
)
def test_files_a_reviewer_would_not_read_are_skipped_with_a_reason(
    entry: dict[str, Any], reason: str
) -> None:
    assert skip_reason(entry) == reason


def test_ordinary_code_and_a_rename_with_changes_are_reviewed() -> None:
    assert skip_reason(changed("src/app.py")) is None
    assert skip_reason(changed("src/app.py", status="renamed", changes=4)) is None
    # A directory named like a vendored one only counts as a directory, not a file name.
    assert skip_reason(changed("src/vendor.py")) is None


def test_code_outranks_tests_and_docs_of_the_same_size() -> None:
    code = parse_patch(PATCH, "src/app.py")[0]
    test = parse_patch(PATCH, "tests/test_app.py")[0]
    doc = parse_patch(PATCH, "docs/guide.md")[0]
    big_test = parse_patch("@@ -1 +1,7 @@\n x\n+1\n+2\n+3\n+4\n+5\n+6\n", "tests/test_big.py")[0]
    assert rank([test, doc, code, big_test]) == [big_test, code, test, doc]


def test_the_listing_numbers_new_file_lines_and_marks_what_is_not_commentable() -> None:
    hunk = parse_patch(PATCH, "src/app.py")[0]
    head = [f"line {n}" for n in range(1, 21)]
    lines = render(hunk, head).splitlines()
    assert lines[0] == "src/app.py"
    assert lines[1] == "    1 . line 1"
    assert "    3       a = 1" in lines
    assert "      -     b = 2" in lines
    assert "    4 +     b = 3" in lines
    assert "    5 +     c = 4" in lines
    assert "    7 . line 7" in lines
    assert lines[-1] == "   16 . line 16"
    assert commentable(hunk) == frozenset({3, 4, 5, 6})


def test_a_head_file_shorter_than_the_hunk_claims_does_not_crash_the_render() -> None:
    hunk = parse_patch(PATCH, "src/app.py")[0]
    assert "line 1" in render(hunk, ["line 1"])


def test_a_huge_hunk_is_cut_rather_than_spending_the_whole_budget() -> None:
    patch = "@@ -0,0 +1,200 @@\n" + "".join(f"+row {n}\n" for n in range(200))
    rendered = render(parse_patch(patch, "big.py")[0])
    assert f"{200 - MAX_HUNK_LINES} more lines of this hunk not shown" in rendered
    assert "row 79" in rendered and "row 80" not in rendered


def test_fit_takes_the_richest_rendering_that_fits_and_drops_in_rank_order() -> None:
    hunk = parse_patch(PATCH, "a.py")[0]
    big, small = "x" * 3000, "y" * 300
    blocks = [Block(hunk, (big, small)), Block(hunk, (big, small)), Block(hunk, (big, small))]
    fitted = fit(blocks, budget=estimate_tokens(big) + estimate_tokens(small))
    assert [text for _, text in fitted.shown] == [big, small]
    assert len(fitted.dropped) == 1
    assert fitted.tokens == estimate_tokens(big) + estimate_tokens(small)


def test_nothing_fitting_is_everything_dropped() -> None:
    hunk = parse_patch(PATCH, "a.py")[0]
    fitted = fit([Block(hunk, ("z" * 900,))], budget=10)
    assert fitted.shown == []
    assert fitted.dropped == [hunk]


@respx.mock
async def test_the_pull_and_its_changes_come_from_the_api(tmp_path: Path) -> None:
    respx.get(f"{API}/repos/o/r/pulls/7").mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Fix run",
                "body": None,
                "user": {"login": "dev"},
                "draft": False,
                "base": {"sha": "b" * 40},
                "head": {"sha": "h" * 40},
            },
        )
    )
    respx.get(f"{API}/repos/o/r/pulls/7/files").mock(
        return_value=httpx.Response(
            200, json=[changed("uv.lock"), changed("tests/test_x.py"), changed("src/x.py")]
        )
    )
    respx.get(f"{API}/repos/o/r/contents/src/x.py").mock(
        return_value=httpx.Response(200, content=b"one\ntwo\n")
    )
    async with httpx.AsyncClient() as http:
        client = GitHubClient(http, "t", HttpCache(tmp_path))
        pull = await fetch_pull(client, "o/r", 7)
        hunks, skipped = await fetch_changes(client, "o/r", 7)
        windows = await fetch_windows(client, "o/r", pull.head_sha, hunks[:1])
    assert (pull.body, pull.author, pull.head_sha) == ("", "dev", "h" * 40)
    assert [h.path for h in hunks] == ["src/x.py", "tests/test_x.py"]
    assert [(s.path, s.reason) for s in skipped] == [("uv.lock", "lockfile or generated")]
    assert windows == {"src/x.py": ["one", "two"]}
