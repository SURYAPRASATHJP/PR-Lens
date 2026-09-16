from pathlib import Path
from typing import Any

import pytest

from pr_lens.corpus.writer import LocalSink, write_units
from pr_lens.eval.corpus import read_corpus
from pr_lens.eval.parts import EMBED_PARTS
from pr_lens.eval.split import HOLDOUT, TUNE, HoldoutViolation
from pr_lens.eval.vectors import corpus_part
from pr_lens.ingest.mine import review_comment_unit
from pr_lens.ingest.units import CorpusUnit
from pr_lens.jobs.embed import embed_input
from pr_lens.retrieval.embed import MODELS
from pr_lens.review.retrieve import MAX_COMMENT_CHARS, load_index
from tests.fakes import BagEncoder

REPO = sorted(TUNE)[0]
MODEL = MODELS["gte-modernbert"]


def comment(comment_id: int, pull: int, body: str, author: str = "reviewer") -> CorpusUnit:
    unit = review_comment_unit(
        REPO,
        {
            "id": comment_id,
            "path": "pkg/cache.py",
            "line": 3,
            "diff_hunk": "@@ -1,2 +1,2 @@\n-a\n+b\n",
            "body": body,
            "user": {"login": author},
            "pull_request_url": f"https://api.github.com/repos/{REPO}/pulls/{pull}",
        },
    )
    assert unit is not None
    return unit


def build(tmp_path: Path, units: list[CorpusUnit]) -> LocalSink:
    sink = LocalSink(tmp_path)
    write_units(sink, REPO, units)
    corpus = read_corpus(sink, REPO)
    encoder = BagEncoder(MODEL)
    for part in range(EMBED_PARTS[MODEL.key]):
        embed_input(sink, corpus_part(corpus, MODEL, part, EMBED_PARTS[MODEL.key]), encoder)
    return sink


@pytest.fixture
def sink(tmp_path: Path) -> LocalSink:
    return build(
        tmp_path,
        [
            comment(1, 10, "The cache eviction here drops the newest entry, not the oldest."),
            comment(2, 20, "Cache eviction again: this loop never frees the oldest entry."),
            comment(3, 30, "Cache eviction is fixed now, thanks. Answer key for pull 30."),
            comment(4, 40, "Unrelated: rename this variable."),
            CorpusUnit(
                repo=REPO,
                kind="source",
                identity="pkg/cache.py#evict#0",
                text="def evict(cache): cache eviction oldest entry",
                path="pkg/cache.py",
            ),
        ],
    )


def test_the_pull_under_review_and_every_later_one_are_never_returned(sink: LocalSink) -> None:
    """The replay leakage rule. Comment 3 is the human's answer on pull 30 itself."""
    index = load_index(sink, REPO, BagEncoder(MODEL))
    [found] = index.search(["cache eviction oldest entry"], before=30)
    numbers = {past.pull_request_number for past in found}
    assert numbers == {10, 20}
    assert all(past.pull_request_number < 30 for past in found)


def test_only_review_comments_are_indexed_and_by_body(sink: LocalSink) -> None:
    index = load_index(sink, REPO, BagEncoder(MODEL))
    assert index.size == 4
    [found] = index.search(["cache eviction oldest entry"], before=1000)
    assert found[0].body.startswith("Cache eviction") or found[0].body.startswith("The cache")
    assert not any(past.body.startswith("@@") for past in found)
    assert found == sorted(found, key=lambda past: -past.score)


def test_an_automated_reviewers_comments_are_not_what_this_project_cares_about(
    tmp_path: Path,
) -> None:
    """Copilot writes a large share of the review comments in some repositories. A draft
    grounded in what reviewers care about should rest on what its people said."""
    sink = build(
        tmp_path,
        [
            comment(1, 5, "cache eviction drops the newest entry", author="maintainer"),
            comment(2, 6, "cache eviction nit from a machine", author="Copilot"),
            comment(3, 7, "cache eviction nit from another machine", author="dependabot[bot]"),
        ],
    )
    index = load_index(sink, REPO, BagEncoder(MODEL))
    assert index.size == 1
    [[past]] = index.search(["cache eviction"], before=100)
    assert past.body.endswith("newest entry")


def test_nothing_before_the_first_pull_is_an_empty_answer(sink: LocalSink) -> None:
    index = load_index(sink, REPO, BagEncoder(MODEL))
    assert index.search(["cache"], before=10) == [[]]
    assert index.search([], before=10) == []


def test_a_long_comment_is_cut(tmp_path: Path) -> None:
    sink = build(tmp_path, [comment(1, 5, "word " * 400)])
    [[past]] = load_index(sink, REPO, BagEncoder(MODEL)).search(["word"], before=6)
    assert len(past.body) <= MAX_COMMENT_CHARS + len(" [cut]")


@pytest.mark.parametrize("repo", sorted(HOLDOUT), ids=str)
def test_every_holdout_repo_is_refused_before_anything_is_read(repo: str) -> None:
    class Untouchable:
        def read_manifest(self, name: str) -> dict[str, str]:
            raise AssertionError("read a holdout manifest")

        def read(self, name: str) -> bytes | None:
            raise AssertionError("read a holdout shard")

        def write(self, files: Any) -> None:
            raise AssertionError("wrote")

    with pytest.raises(HoldoutViolation):
        load_index(Untouchable(), repo, BagEncoder(MODEL))
    with pytest.raises(HoldoutViolation):
        load_index(Untouchable(), repo.upper(), BagEncoder(MODEL))
