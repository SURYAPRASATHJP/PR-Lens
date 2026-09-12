"""Embed one slice of the tune corpus, or of the frozen query set, into the dataset repo.

One slice per invocation so the workflow can fan the work out as a matrix. The long-
context model on a 4 vCPU runner is the slow part of Phase 2 by a wide margin, and a
matrix turns a job that would run past the six hour cap into many that finish well inside
it. Twenty run at once on a free account.

The skip reuses Phase 1's content hashing rather than inventing a second scheme. A repo's
corpus fingerprint is a digest of its shard manifest, which is the hash of every shard, so
a re-run over an unchanged corpus reads a few bytes per part and embeds nothing.
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence

from pr_lens.eval.corpus import load_repo
from pr_lens.eval.split import require_tune
from pr_lens.eval.store import Store, build_store
from pr_lens.eval.vectors import PartInput, corpus_part, query_part
from pr_lens.jobs.pairs import load_pairs
from pr_lens.logging import configure
from pr_lens.retrieval.embed import MODELS, EmbedReport, Encoder, SentenceEncoder, embed_part

logger = logging.getLogger(__name__)


def embed_input(store: Store, target: PartInput, encoder: Encoder) -> EmbedReport:
    return embed_part(store, target.path, target.fingerprint, target.ids, target.texts, encoder)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-embed", description=__doc__)
    parser.add_argument("--target", choices=("corpus", "queries"), required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--repo", help="the tune repo to embed, for --target corpus")
    parser.add_argument("--version", help="the frozen query set, for --target queries")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--sink", choices=("local", "huggingface"), default="local")
    parser.add_argument("--corpus-dir", default=".corpus")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    store = build_store(args.sink, args.corpus_dir)
    model = MODELS[args.model]

    if args.target == "corpus":
        if not args.repo:
            logger.error("--target corpus needs --repo")
            return 1
        (repo,) = require_tune([args.repo])
        target = corpus_part(load_repo(store, repo), model, args.part, args.parts)
    else:
        if not args.version:
            logger.error("--target queries needs --version")
            return 1
        pairs, manifest = load_pairs(store, args.version)
        target = query_part(
            pairs, str(manifest["sha256"]), args.version, model, args.part, args.parts
        )

    embed_input(store, target, SentenceEncoder(model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
