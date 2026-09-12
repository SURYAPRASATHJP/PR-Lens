"""A frozen query set stays frozen: never overwritten, and refused if it was altered."""

from pathlib import Path

import pytest

from pr_lens.corpus.writer import LocalSink
from pr_lens.eval.pairs import RepoPairs, ReviewPair, pairs_path
from pr_lens.jobs.pairs import FrozenPairsError, freeze, load_pairs, parse_args, run


def pair(comment_id: int) -> ReviewPair:
    return ReviewPair(
        repo="encode/httpx",
        comment_id=comment_id,
        pull_request=1,
        path="httpx/_client.py",
        diff_hunk="@@ -1 +1 @@\n-timeout = 5\n+timeout = None",
        body="A None timeout hangs forever on a dead server, keep a default.",
        author="reviewer",
        created_at="",
        html_url="",
    )


def test_a_frozen_set_reads_back_exactly(tmp_path: Path) -> None:
    sink = LocalSink(tmp_path)
    manifest = freeze(
        sink, "v1", [RepoPairs(repo="encode/httpx", seen=2, pairs=[pair(1), pair(2)])]
    )
    pairs, stored = load_pairs(sink, "v1")
    assert [p.comment_id for p in pairs] == [1, 2]
    assert stored["sha256"] == manifest["sha256"]


def test_freezing_is_deterministic(tmp_path: Path) -> None:
    first = freeze(
        LocalSink(tmp_path / "a"), "v1", [RepoPairs("encode/httpx", 2, [pair(2), pair(1)])]
    )
    second = freeze(
        LocalSink(tmp_path / "b"), "v1", [RepoPairs("encode/httpx", 2, [pair(1), pair(2)])]
    )
    assert first["sha256"] == second["sha256"]


def test_an_altered_pair_file_is_refused(tmp_path: Path) -> None:
    sink = LocalSink(tmp_path)
    freeze(sink, "v1", [RepoPairs(repo="encode/httpx", seen=1, pairs=[pair(1)])])
    sink.write({pairs_path("v1"): (tmp_path / pairs_path("v1")).read_bytes() + b"\x00"})
    with pytest.raises(FrozenPairsError):
        load_pairs(sink, "v1")


async def test_a_second_run_of_a_frozen_version_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = LocalSink(tmp_path)
    freeze(sink, "v1", [RepoPairs(repo="encode/httpx", seen=1, pairs=[pair(1)])])
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    # No mining token: if the job tried to mine again it would fail rather than skip.
    monkeypatch.delenv("GH_MINING_TOKEN", raising=False)
    monkeypatch.delenv("GH_DISPATCH_TOKEN", raising=False)

    assert await run(parse_args(["--version", "v1", "--corpus-dir", str(tmp_path)])) == 0
    after = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
