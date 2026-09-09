"""Shard writing has to be byte-deterministic, or nothing downstream can be skipped."""

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pr_lens.corpus.writer import LocalSink, write_units
from pr_lens.ingest.units import CorpusUnit


def units(count: int, *, text: str = "body", repo: str = "octocat/hello-world") -> list[CorpusUnit]:
    return [
        CorpusUnit(
            repo=repo,
            kind="source" if index % 2 == 0 else "doc",
            identity=f"file{index}.py##0",
            text=f"{text} {index}",
            path=f"file{index}.py",
        )
        for index in range(count)
    ]


def test_shards_are_grouped_by_kind_and_named_after_the_repo(tmp_path: Path) -> None:
    write_units(LocalSink(tmp_path), "octocat/hello-world", units(4))

    assert (tmp_path / "source" / "octocat__hello-world-00000.jsonl.gz").exists()
    assert (tmp_path / "doc" / "octocat__hello-world-00000.jsonl.gz").exists()


def test_the_same_units_produce_byte_identical_shards(tmp_path: Path) -> None:
    # gzip stamps the clock into its header unless told otherwise, which would make every
    # shard differ from the last one and force a full re-upload every night.
    first = tmp_path / "one"
    second = tmp_path / "two"
    write_units(LocalSink(first), "octocat/hello-world", units(6))
    write_units(LocalSink(second), "octocat/hello-world", list(reversed(units(6))))

    for path in first.rglob("*.gz"):
        assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()


def test_a_second_write_of_unchanged_units_sends_nothing(tmp_path: Path) -> None:
    sink = LocalSink(tmp_path)
    write_units(sink, "octocat/hello-world", units(6))

    report = write_units(sink, "octocat/hello-world", units(6))

    assert report.written == ()
    assert len(report.unchanged) == 2


def test_changed_text_sends_only_the_shard_that_changed(tmp_path: Path) -> None:
    sink = LocalSink(tmp_path)
    write_units(sink, "octocat/hello-world", units(6))

    changed = units(6)
    changed[0] = replace(changed[0], text="edited")
    report = write_units(sink, "octocat/hello-world", changed)

    assert [name.split("/")[0] for name in report.written] == ["source"]
    assert len(report.unchanged) == 1


def test_the_manifest_records_every_shard(tmp_path: Path) -> None:
    write_units(LocalSink(tmp_path), "octocat/hello-world", units(4))

    manifest = json.loads(
        (tmp_path / "manifests" / "octocat__hello-world.json").read_text(encoding="utf-8")
    )
    assert set(manifest) == {
        "source/octocat__hello-world-00000.jsonl.gz",
        "doc/octocat__hello-world-00000.jsonl.gz",
    }


def test_a_shard_holds_one_json_record_per_line_carrying_the_text(tmp_path: Path) -> None:
    write_units(LocalSink(tmp_path), "octocat/hello-world", units(2))

    shard = tmp_path / "source" / "octocat__hello-world-00000.jsonl.gz"
    records = [json.loads(line) for line in gzip.decompress(shard.read_bytes()).splitlines()]
    assert len(records) == 1
    assert records[0]["text"] == "body 0"
    assert records[0]["unit_id"]


def test_every_unit_is_placed_in_the_shard_that_holds_it(tmp_path: Path) -> None:
    written = units(6)
    report = write_units(LocalSink(tmp_path), "octocat/hello-world", written)

    assert set(report.placement) == {u.unit_id for u in written}
    for unit in written:
        assert report.placement[unit.unit_id].startswith(f"{unit.kind}/")


def test_a_shard_rolls_over_at_the_row_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pr_lens.corpus.writer.MAX_SHARD_ROWS", 2)
    report = write_units(LocalSink(tmp_path), "octocat/hello-world", units(8))

    assert sorted(report.written) == [
        "doc/octocat__hello-world-00000.jsonl.gz",
        "doc/octocat__hello-world-00001.jsonl.gz",
        "source/octocat__hello-world-00000.jsonl.gz",
        "source/octocat__hello-world-00001.jsonl.gz",
    ]
    assert report.rows == 8


def test_the_row_count_is_the_number_of_units(tmp_path: Path) -> None:
    assert write_units(LocalSink(tmp_path), "octocat/hello-world", units(7)).rows == 7
