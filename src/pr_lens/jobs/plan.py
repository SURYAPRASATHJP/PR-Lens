"""How Phase 2's work is split across runners, written down once.

The workflow builds its matrices by running this module, and the recall job reassembles
the parts with the same numbers, so the two cannot disagree about how many parts a repo
was cut into. Sized from CPU throughput measured 12 Sep 2026 at four threads: bge-small
about 11 to 15 texts a second, gte-modernbert 0.45 to 4 depending on length, and the
reranker 0.6 pairs a second. Every matrix job is meant to finish in well under an hour
against a six hour cap.
"""

import json
import sys
from collections.abc import Callable, Sequence

from pr_lens.eval.split import tune_repos

PAIRS_VERSION = "v1"

EMBED_PARTS = {"bge-small": 1, "gte-modernbert": 8}
QUERY_PARTS = {"bge-small": 1, "gte-modernbert": 4}

# Pairs reranked, each under both query variants, so twice this many rerank passes of 30
# candidates. 300 puts the 95% interval on a recall near 0.5 at about 0.06, enough to see
# where the curve flattens, and the depths are compared on the same queries.
RERANK_SAMPLE = 300
RERANK_SHARDS = 12

ANCHOR_PARTS = 8


def embed_matrix() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for model, parts in EMBED_PARTS.items():
        for repo in tune_repos():
            for part in range(parts):
                jobs.append(
                    {"target": "corpus", "model": model, "repo": repo, "part": part, "parts": parts}
                )
    for model, parts in QUERY_PARTS.items():
        for part in range(parts):
            jobs.append(
                {"target": "queries", "model": model, "repo": "", "part": part, "parts": parts}
            )
    return jobs


def rerank_matrix() -> list[dict[str, object]]:
    return [{"shard": shard, "shards": RERANK_SHARDS} for shard in range(RERANK_SHARDS)]


def anchor_matrix() -> list[dict[str, object]]:
    return [{"part": part, "parts": ANCHOR_PARTS} for part in range(ANCHOR_PARTS)]


MATRICES: dict[str, Callable[[], list[dict[str, object]]]] = {
    "embed": embed_matrix,
    "rerank": rerank_matrix,
    "anchor": anchor_matrix,
}


def main(argv: Sequence[str] | None = None) -> int:
    which = (argv if argv is not None else sys.argv[1:])[0]
    sys.stdout.write(json.dumps({"include": MATRICES[which]()}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
