"""The identity and hash rules the idempotence gate depends on."""

from pr_lens.ingest.units import CorpusUnit


def unit(**overrides: object) -> CorpusUnit:
    base: dict[str, object] = {
        "repo": "octocat/hello-world",
        "kind": "source",
        "identity": "src/thing.py#parse#0",
        "text": "def parse():\n    return 1\n",
        "ref": "a" * 40,
        "path": "src/thing.py",
        "symbol": "parse",
        "start_line": 10,
        "end_line": 11,
    }
    return CorpusUnit(**{**base, **overrides})  # type: ignore[arg-type]


def test_the_same_unit_hashes_the_same_way_twice() -> None:
    assert unit().unit_id == unit().unit_id
    assert unit().content_hash == unit().content_hash


def test_a_new_commit_alone_changes_neither_the_id_nor_the_hash() -> None:
    # This is the whole gate. If ref reached either hash, every push to a mined repo
    # would rewrite that repo's entire corpus and re-ingest would never converge.
    moved = unit(ref="b" * 40)
    assert moved.unit_id == unit().unit_id
    assert moved.content_hash == unit().content_hash


def test_edited_text_changes_the_hash_but_not_the_id() -> None:
    edited = unit(text="def parse():\n    return 2\n")
    assert edited.unit_id == unit().unit_id
    assert edited.content_hash != unit().content_hash


def test_a_chunk_that_slid_down_the_file_changes_the_hash() -> None:
    # A stale line number is a citation pointing at the wrong code, so it counts as a
    # change even though the text is identical.
    assert unit(start_line=40, end_line=41).content_hash != unit().content_hash


def test_a_renamed_symbol_is_a_different_unit() -> None:
    assert unit(identity="src/thing.py#parse_all#0").unit_id != unit().unit_id


def test_the_same_identity_in_two_repos_is_two_units() -> None:
    assert unit(repo="other/repo").unit_id != unit().unit_id


def test_metadata_is_part_of_the_content() -> None:
    titled = unit(kind="pull_request", metadata={"title": "Fix the parser"})
    retitled = unit(kind="pull_request", metadata={"title": "Fix the parser again"})

    assert titled.content_hash != retitled.content_hash


def test_the_hash_covers_exactly_the_shard_record() -> None:
    """The invariant the shard-skipping rests on.

    write_units skips uploading a shard whose bytes match the manifest. That is only
    correct if two units with the same hash serialise identically, so the hash is taken
    over the record itself rather than over a chosen subset of its fields.
    """
    import hashlib
    import json

    record = dict(unit().as_record())
    stated = record.pop("content_hash")
    recomputed = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()

    assert stated == recomputed


def test_the_postgres_row_carries_no_text_and_the_shard_record_does() -> None:
    # Neon free is 0.5 GB. Corpus text lives in the dataset repo, not in the database.
    assert "text" not in unit().as_row()
    assert unit().as_record()["text"] == unit().text
    assert unit().as_row()["char_count"] == len(unit().text)


def test_the_shard_record_carries_no_commit_sha() -> None:
    # ref moves on every push without the content moving. In the record it would make
    # every shard differ after any push, and re-upload the whole corpus to say nothing.
    assert "ref" not in unit().as_record()
    assert unit().as_row()["ref"] == "a" * 40
