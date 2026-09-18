from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient
from pr_lens.ingest.diff import parse_patch
from pr_lens.review.context import (
    IMPORTS_HEADING,
    MAX_HUNK_LINES,
    MAX_IMPORT_LINES_PER_FILE,
    Block,
    commentable,
    fetch_changes,
    fetch_pull,
    fetch_windows,
    fit,
    imports,
    rank,
    render,
    render_imports,
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


def test_the_code_of_every_hunk_is_fitted_before_any_hunk_is_enriched() -> None:
    """The defect that made the first live comment false, as an assertion.

    Under the old single greedy pass the first block took its rich rendering, the budget
    ran out, and the two below it were dropped whole. Their code costs a fraction of one
    window, so all three belong in the prompt and the enrichment is what gives way.
    """
    hunk = parse_patch(PATCH, "a.py")[0]
    rich, bare = "x" * 3000, "y" * 300
    blocks = [Block(hunk, (rich, bare))] * 3
    fitted = fit(blocks, budget=estimate_tokens(rich) + estimate_tokens(bare))
    assert [text for _, text in fitted.shown] == [bare, bare, bare]
    assert fitted.dropped == []


def test_a_hunk_too_large_to_fit_does_not_drop_the_ones_ranked_below_it() -> None:
    """Pull request 16 in one line: ten one line hunks dropped behind a hunk that did not
    fit, one of them the import that decided whether the posted comment was true."""
    hunk = parse_patch(PATCH, "a.py")[0]
    huge, small = "x" * 9000, "y" * 200
    blocks = [Block(hunk, (small,)), Block(hunk, (huge,)), Block(hunk, (small,))]
    fitted = fit(blocks, budget=estimate_tokens(small) * 2 + 5)
    assert [text for _, text in fitted.shown] == [small, small]
    assert fitted.dropped == [hunk]
    assert fitted.positions == (0, 2)


def test_what_is_left_after_the_code_upgrades_the_best_ranked_hunk_first() -> None:
    hunk = parse_patch(PATCH, "a.py")[0]
    rich, bare = "x" * 1200, "y" * 200
    blocks = [Block(hunk, (rich, bare)), Block(hunk, (rich, bare))]
    fitted = fit(blocks, budget=estimate_tokens(rich) + estimate_tokens(bare))
    assert [text for _, text in fitted.shown] == [rich, bare]
    assert fitted.tokens == estimate_tokens(rich) + estimate_tokens(bare)


def test_nothing_fitting_is_everything_dropped_whatever_the_rank() -> None:
    hunk = parse_patch(PATCH, "a.py")[0]
    fitted = fit([Block(hunk, ("z" * 900,)), Block(hunk, ("z" * 900,))], budget=10)
    assert fitted.shown == []
    assert fitted.dropped == [hunk, hunk]
    assert fitted.positions == ()


PY_HEAD = [
    '"""A module."""',
    "",
    "import os",
    "from pkg.types import (",
    "    _CliVariadicArg,",
    "    Other,",
    ")",
    "",
    "def run():",
    "    import json",
    "    return json, os",
]


def test_a_parenthesised_import_keeps_the_names_on_its_continuation_lines() -> None:
    """The names are on the continuation lines. Taking only the first line would list the
    module and hide every name it brings in, which is the whole point of showing it."""
    found = imports(PY_HEAD)
    assert found == [
        (3, "import os"),
        (4, "from pkg.types import ("),
        (5, "    _CliVariadicArg,"),
        (6, "    Other,"),
        (7, ")"),
    ]


def test_an_import_inside_a_function_is_not_a_name_in_scope_at_module_level() -> None:
    """Line 10 is `    import json`, indented inside run(). Asserted by line number: the
    first version of this test compared the text, which still passed when the indentation
    check was removed, because the captured text keeps its leading spaces."""
    assert 10 not in [number for number, _ in imports(PY_HEAD)]
    assert [number for number, _ in imports(PY_HEAD)] == [3, 4, 5, 6, 7]


def test_javascript_imports_re_exports_and_requires_are_found() -> None:
    lines = [
        "import { parse } from './parse';",
        "export { Thing } from './thing';",
        "const fs = require('fs');",
        "export function go() {}",
        "  import sneaky from 'x';",
    ]
    assert [number for number, _ in imports(lines)] == [1, 2, 3]


def test_one_file_cannot_spend_the_whole_import_allowance() -> None:
    lines = [f"import m{n}" for n in range(MAX_IMPORT_LINES_PER_FILE + 20)]
    assert len(imports(lines)) == MAX_IMPORT_LINES_PER_FILE


def test_the_import_listing_names_its_file_and_marks_its_rows_uncommentable() -> None:
    rendered = render_imports({"src/app.py": PY_HEAD})
    assert rendered.startswith(IMPORTS_HEADING)
    assert "src/app.py" in rendered
    assert "    3 . import os" in rendered
    assert "    5 .     _CliVariadicArg," in rendered
    # The hunk markers say "+" and " " are commentable. These rows must not claim to be.
    assert " + " not in rendered


def test_a_changed_file_with_no_imports_adds_no_section() -> None:
    assert render_imports({"docs/guide.md": ["# Title", "Some prose."]}) == ""
    assert render_imports({}) == ""


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
