import json
from pathlib import Path
from typing import Any

import pytest

from pr_lens.corpus.writer import LocalSink, write_units
from pr_lens.eval.corpus import body_only, load_repo
from pr_lens.eval.split import HoldoutViolation
from pr_lens.ingest.chunking import MAX_CHARS
from pr_lens.ingest.mine import review_comment_unit
from pr_lens.ingest.units import CorpusUnit

REPO = "fastapi/typer"


def payload(**overrides: Any) -> dict[str, Any]:
    return {
        "id": 42,
        "path": "typer/main.py",
        "line": 12,
        "diff_hunk": "@@ -10,3 +10,3 @@ def main():\n     x = 1\n-    y = '/'\n+    y = os.sep\n",
        "body": "  This breaks on Windows, os.sep is a backslash there.\n\nSecond paragraph.  ",
        "user": {"login": "reviewer"},
        "pull_request_url": "https://api.github.com/repos/fastapi/typer/pulls/7",
        **overrides,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"line": None, "original_line": None},
        {"path": None},
        {"diff_hunk": ""},
        {"body": "Quoting the hunk further down is fine:\n\n     x = 1\n-    y = '/'"},
        {"body": "こんにちは、この変更で全角スペースが消えます。テストも直してください。"},
    ],
    ids=["ordinary", "no line", "no path", "no hunk", "quotes hunk later", "unicode"],
)
def test_body_only_recovers_exactly_the_body(overrides: dict[str, Any]) -> None:
    comment = payload(**overrides)
    unit = review_comment_unit(REPO, comment)
    assert unit is not None
    assert body_only(unit.as_record()) == comment["body"].strip()


def test_the_with_hunk_text_really_does_contain_the_hunk() -> None:
    """The leak the body-only row removes. If this stops being true, the two rows mean
    nothing, and the review_comment layout changed without the eval noticing."""
    comment = payload()
    unit = review_comment_unit(REPO, comment)
    assert unit is not None
    assert comment["diff_hunk"].strip() in unit.text
    assert comment["diff_hunk"].strip() not in body_only(unit.as_record())


def source(identity: str, text: str) -> CorpusUnit:
    return CorpusUnit(repo=REPO, kind="source", identity=identity, text=text, path="a.py")


def test_load_repo_reads_back_what_phase_1_wrote(tmp_path: Path) -> None:
    comment = review_comment_unit(REPO, payload())
    assert comment is not None
    units = [
        source("a.py#exact#0", "x" * MAX_CHARS),
        source("a.py#unicode#0", "def 挨拶():\n    return '👋🏽 ' * 200\n" * 20),
        comment,
    ]
    sink = LocalSink(tmp_path)
    write_units(sink, REPO, units)

    corpus = load_repo(sink, REPO)

    # No issue shard at all, like the repo that returned no issues inside the window.
    assert {d.kind for d in corpus.documents} == {"source", "review_comment"}
    by_id = {d.unit_id: d for d in corpus.documents}
    assert len(by_id[units[0].unit_id].text) == MAX_CHARS
    assert by_id[units[1].unit_id].text == units[1].text
    loaded_comment = by_id[comment.unit_id]
    assert loaded_comment.serialised("with_hunk") == comment.text
    assert loaded_comment.serialised("body_only") == payload()["body"].strip()
    assert by_id[units[0].unit_id].serialised("body_only") == units[0].text
    assert [d.unit_id for d in corpus.documents] == sorted(by_id)


def test_the_fingerprint_moves_with_the_corpus_and_only_with_it(tmp_path: Path) -> None:
    sink = LocalSink(tmp_path)
    write_units(sink, REPO, [source("a.py#f#0", "def f(): pass")])
    first = load_repo(sink, REPO).fingerprint
    write_units(sink, REPO, [source("a.py#f#0", "def f(): pass")])
    assert load_repo(sink, REPO).fingerprint == first
    write_units(sink, REPO, [source("a.py#f#0", "def f(): return 1")])
    assert load_repo(sink, REPO).fingerprint != first


def test_a_missing_shard_is_an_error_not_a_smaller_corpus(tmp_path: Path) -> None:
    sink = LocalSink(tmp_path)
    write_units(sink, REPO, [source("a.py#f#0", "def f(): pass")])
    manifest = json.loads((tmp_path / "manifests" / "fastapi__typer.json").read_text())
    for shard in manifest:
        (tmp_path / shard).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_repo(sink, REPO)


def test_a_holdout_repo_is_never_read(tmp_path: Path) -> None:
    with pytest.raises(HoldoutViolation):
        load_repo(LocalSink(tmp_path), "pallets/click")
