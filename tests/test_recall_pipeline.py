"""The three recall steps end to end, on a synthetic corpus, with the models faked.

This is the wiring: frozen pairs in, vectors per part, first stage, rerank shards, table
out. A run on Actions takes hours of runner time, and a mismatch between the parts one
job wrote and the parts the next expects should cost seconds here instead.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pr_lens.corpus.writer import LocalSink, write_units
from pr_lens.eval import split
from pr_lens.eval.corpus import load_repo
from pr_lens.eval.pairs import RepoPairs, ReviewPair
from pr_lens.eval.split import HoldoutViolation, tune_repos
from pr_lens.eval.vectors import corpus_part, query_part
from pr_lens.ingest.mine import review_comment_unit
from pr_lens.ingest.units import CorpusUnit
from pr_lens.jobs import recall
from pr_lens.jobs.embed import embed_input
from pr_lens.jobs.pairs import freeze, load_pairs
from pr_lens.jobs.plan import EMBED_PARTS, QUERY_PARTS
from pr_lens.retrieval.embed import MODELS, EmbeddingModel

VERSION = "test"


class BagEncoder:
    """Hashed bag of words, so a query and a document sharing words land close together."""

    def __init__(self, model: EmbeddingModel, batch_size: int = 16) -> None:
        del batch_size
        self.model = model

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        rows = np.zeros((len(texts), self.model.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.model.dims
                rows[i, slot] += 1.0
            norm = np.linalg.norm(rows[i])
            rows[i] = rows[i] / norm if norm else rows[i]
            if not norm:
                rows[i, 0] = 1.0
        return rows

    def token_counts(self, texts: Sequence[str]) -> npt.NDArray[np.int32]:
        return np.asarray([len(t.split()) * 3 for t in texts], dtype=np.int32)


class WordOverlapReranker:
    def __init__(self, max_tokens: int, batch_size: int = 8) -> None:
        del max_tokens, batch_size

    def score(self, query: str, documents: Sequence[str]) -> npt.NDArray[np.float32]:
        words = set(re.findall(r"\w+", query.lower()))
        return np.asarray(
            [len(words & set(re.findall(r"\w+", d.lower()))) for d in documents],
            dtype=np.float32,
        )


def build_corpus(sink: LocalSink) -> list[ReviewPair]:
    pairs = []
    for r, repo in enumerate(tune_repos()):
        units: list[CorpusUnit] = []
        for i in range(6):
            topic = f"topic{r}x{i}"
            units.append(
                CorpusUnit(
                    repo=repo,
                    kind="source",
                    identity=f"pkg/mod{i}.py#f#0",
                    # Its own vocabulary, so that a dense retriever that cannot find the
                    # gold is a misaligned index rather than a crowded one.
                    text=f"class Registry{r}Slot{i}:\n    entries = []\n",
                    path=f"pkg/mod{i}.py",
                )
            )
            comment_id = 1000 * r + i
            hunk = (
                f"@@ -1,3 +1,3 @@ def handler_{topic}(value):\n"
                f"     cache_{topic} = load()\n"
                f"-    return compute_{topic}(value)\n"
                f"+    return compute_{topic}(value, strict=True)"
            )
            body = f"Passing strict here changes compute_{topic} for every caller of cache_{topic}."
            payload = {
                "id": comment_id,
                "path": f"pkg/mod{i}.py",
                "line": 3,
                "diff_hunk": hunk,
                "body": body,
                "pull_request_url": f"https://api.github.com/repos/{repo}/pulls/{i + 1}",
            }
            unit = review_comment_unit(repo, payload)
            assert unit is not None
            units.append(unit)
            pairs.append(
                ReviewPair(
                    repo=repo,
                    comment_id=comment_id,
                    pull_request=i + 1,
                    path=f"pkg/mod{i}.py",
                    diff_hunk=hunk,
                    body=body,
                    author="reviewer",
                    created_at="",
                    html_url="",
                )
            )
        write_units(sink, repo, units)
    return pairs


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalSink:
    monkeypatch.setattr(recall, "SentenceEncoder", BagEncoder)
    monkeypatch.setattr(recall, "CrossEncoderReranker", WordOverlapReranker)
    monkeypatch.setattr(recall, "RERANK_SAMPLE", 40)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    sink = LocalSink(tmp_path)
    pairs = build_corpus(sink)
    by_repo = {repo: RepoPairs(repo=repo, seen=6) for repo in tune_repos()}
    for pair in pairs:
        by_repo[pair.repo].pairs.append(pair)
    freeze(sink, VERSION, list(by_repo.values()))

    frozen, manifest = load_pairs(sink, VERSION)
    for key, model in MODELS.items():
        encoder = BagEncoder(model)
        for repo in tune_repos():
            corpus = load_repo(sink, repo)
            for part in range(EMBED_PARTS[key]):
                embed_input(sink, corpus_part(corpus, model, part, EMBED_PARTS[key]), encoder)
        for part in range(QUERY_PARTS[key]):
            target = query_part(
                frozen, str(manifest["sha256"]), VERSION, model, part, QUERY_PARTS[key]
            )
            embed_input(sink, target, encoder)
    return sink


def test_the_three_steps_produce_a_complete_table(store: LocalSink) -> None:
    first = recall.first_stage(store, VERSION)
    assert first["corpus"]["repos"] == 18
    assert {r["retriever"] for r in first["first_stage"]} == {"bm25", "dense", "rrf"}
    assert all(0.0 <= v <= 1.0 for r in first["first_stage"] for v in r["recall"].values())

    for shard in range(2):
        recall.rerank(store, VERSION, shard, 2)
    markdown = recall.table(store, VERSION, 2)

    for heading in (
        "## 1. First stage",
        "## 2. Leakage",
        "## 3. The truncation price",
        "## 4. Reranker plateau",
        "## 5. Post-filter recall collapse",
        "## 6. Latency",
        "## 7. The query set",
    ):
        assert heading in markdown
    assert "Not computed yet" not in markdown
    assert (Path(store.root) / "eval/results/test/recall-table.md").read_text() == markdown

    # The body-only rows cannot be string matching, and the with-hunk rows are: the gold
    # carries the query hunk verbatim, so BM25 finds it first every time.
    with_hunk_bm25 = next(
        r
        for r in first["first_stage"]
        if r["retriever"] == "bm25"
        and r["serialisation"] == "with_hunk"
        and r["variant"] == "reviewed"
    )
    assert with_hunk_bm25["recall"]["1"] == 1.0

    # Body-only dense retrieval finds the gold through the shared words. If the body vectors
    # were ever assembled out of order against the documents, this is what would drop.
    body_dense = [
        r
        for r in first["first_stage"]
        if r["retriever"] == "dense" and r["serialisation"] == "body_only"
    ]
    assert body_dense
    assert all(r["recall"]["10"] > 0.9 for r in body_dense)


def test_rerank_shards_survive_a_rerun_of_the_first_stage_but_not_new_candidates(
    store: LocalSink,
) -> None:
    recall.first_stage(store, VERSION)
    recall.rerank(store, VERSION, 0, 1)
    # A rerun over the same inputs has different latency numbers and the same candidates,
    # so the shard is still valid.
    recall.first_stage(store, VERSION)
    recall.table(store, VERSION, 1)

    path = Path(store.root) / "eval/results/test/first-stage.json"
    first = json.loads(path.read_text())
    first["rerank"]["items"][0]["candidates"].reverse()
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="older first stage"):
        recall.table(store, VERSION, 1)


def test_the_harness_refuses_a_holdout_repo_in_its_own_list(
    store: LocalSink, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half of the split rule that matters: what the harness indexes is a subset of
    TUNE, however its repo list is produced."""
    monkeypatch.setattr(recall, "tune_repos", lambda: [*split.tune_repos(), "pallets/click"])
    with pytest.raises(HoldoutViolation, match="holdout"):
        recall.first_stage(store, VERSION)
