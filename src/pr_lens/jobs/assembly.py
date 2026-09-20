"""The assembly table over the tune set's review pairs, with no model call at all.

`eval/assembly.py` has had its answer since 16 Sep and nothing ran it. This is the runner.
Every pull request that carries a frozen query-set pair goes through the same
`fetch_changes`, the same `rank`, the same `MAX_CANDIDATE_HUNKS` cut and the same `fit`
that `review` uses, and then each human comment is asked where it ended up.

The question it exists to answer: batch 2026-09-18-b lost 263 hunks to the candidate
ceiling against 16 to the token budget, which means `MAX_CANDIDATE_HUNKS` and not the
budget is what now decides what the reviewer sees. Whether that matters depends entirely
on one thing, and only the golden set knows it: **does the hunk a human actually commented
on ever rank past the eighth?** If it never does, the ceiling is free and raising it buys
nothing but tokens. If it does, the rank distribution says where to put it.

So the table reports `unconsidered` as its own column rather than folded into `dropped`,
and the rank histogram underneath it is the part that decides anything.

No inference, no database. The GitHub reads are the Phase 1 client's, so a second run is
mostly cache. Safe to run while the day's model tokens are spent.
"""

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence

import httpx

from pr_lens.eval.assembly import (
    Assembly,
    Gold,
    Outcome,
    classify,
    commented_line,
    header,
    rank_of,
    summarise,
)
from pr_lens.eval.pairs import ReviewPair
from pr_lens.eval.split import require_tune, tune_repos
from pr_lens.eval.store import Store, build_store
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import GitHubClient, GitHubError
from pr_lens.github.mining import DEFAULT_CACHE_DIR, mining_token
from pr_lens.jobs.pairs import load_pairs
from pr_lens.jobs.plan import PAIRS_VERSION
from pr_lens.logging import configure
from pr_lens.retrieval.embed import MODELS, Encoder, SentenceEncoder
from pr_lens.review.context import Block, fetch_changes, fetch_pull, fetch_windows, fit
from pr_lens.review.pipeline import MAX_CANDIDATE_HUNKS, draft_frame
from pr_lens.review.prompts import block_variants
from pr_lens.review.retrieve import CommentIndex, load_index

logger = logging.getLogger(__name__)


async def measure_pull(
    client: GitHubClient,
    index: CommentIndex,
    repo: str,
    number: int,
    golds: Sequence[Gold],
) -> tuple[list[Outcome], list[int | None]]:
    """One pull request assembled exactly as `review` would, then asked where each gold went.

    Returns an outcome per gold and the rank of the hunk holding it, counting from one,
    which is None when no hunk holds it at all.
    """
    pull = await fetch_pull(client, repo, number)
    hunks, skipped = await fetch_changes(client, repo, number)
    candidates = list(hunks[:MAX_CANDIDATE_HUNKS])
    unconsidered = list(hunks[MAX_CANDIDATE_HUNKS:])
    if not candidates:
        return [Outcome.NOT_IN_DIFF for _ in golds], [rank_of(g, hunks) for g in golds]

    windows = await fetch_windows(client, repo, pull.head_sha, candidates)
    # `before` is the pull request's own number, the same cutoff replay uses, so a comment
    # cannot retrieve itself or anything written after the code under review.
    past = index.search([hunk.text for hunk in candidates], before=number)
    _, _, budget = draft_frame(pull, windows)
    blocks = [
        Block(hunk, block_variants(n, hunk, windows.get(hunk.path), past[n - 1]))
        for n, hunk in enumerate(candidates, start=1)
    ]
    fitted = fit(blocks, budget)
    return (
        [classify(gold, fitted, skipped, unconsidered) for gold in golds],
        [rank_of(gold, hunks) for gold in golds],
    )


def golds_by_pull(
    pairs: Sequence[ReviewPair],
) -> dict[tuple[str, int], list[tuple[ReviewPair, Gold]]]:
    """The pairs whose commented line can be recovered, grouped by pull request.

    A pair whose diff_hunk will not parse has no gold location and is counted as skipped
    input rather than silently scored, because a pair that cannot be placed would
    otherwise read as NOT_IN_DIFF and flatter the table.
    """
    grouped: dict[tuple[str, int], list[tuple[ReviewPair, Gold]]] = defaultdict(list)
    for pair in pairs:
        line = commented_line(pair.diff_hunk, pair.path)
        if line is None:
            continue
        grouped[(pair.repo, pair.pull_request)].append((pair, Gold(pair.path, line)))
    return grouped


def render(
    overall: Assembly, per_repo: dict[str, Assembly], ranks: Counter[int], unplaceable: int
) -> str:
    lines = [
        "## Assembly, the tune set's review pairs",
        "",
        f"Candidate ceiling {MAX_CANDIDATE_HUNKS} hunks. No model call; the same rank, "
        "cut and fit the pipeline uses.",
        "",
        header(),
        overall.row(),
        "",
        "### By repository",
        "",
        header("repo"),
    ]
    lines.extend(row.row(repo) for repo, row in sorted(per_repo.items()))

    placed = sum(ranks.values())
    past = sum(count for rank, count in ranks.items() if rank > MAX_CANDIDATE_HUNKS)
    lines += [
        "",
        "### Where the commented hunk ranked",
        "",
        "The question the ceiling turns on. A gold hunk ranked past the cut was never "
        "costed against the budget, so raising the budget could not have reached it.",
        "",
        "| rank | gold hunks | cumulative |",
        "|---|---|---|",
    ]
    running = 0
    for rank in sorted(ranks):
        running += ranks[rank]
        label = str(rank) if rank <= MAX_CANDIDATE_HUNKS else f"{rank} (past the cut)"
        lines.append(f"| {label} | {ranks[rank]} | {running / placed:.3f} |" if placed else "")
    share = past / placed if placed else 0.0
    lines += [
        "",
        f"{past} of {placed} placed gold hunks ranked past {MAX_CANDIDATE_HUNKS}, "
        f"{share:.1%}. {unplaceable} pairs could not be placed and are excluded.",
    ]
    return "\n".join(lines) + "\n"


async def run(store: Store, encoder: Encoder, repos: Sequence[str], limit: int) -> str:
    pairs, _ = load_pairs(store, PAIRS_VERSION)
    wanted = {repo.lower() for repo in require_tune(repos)}
    mine = [pair for pair in pairs if pair.repo.lower() in wanted]
    grouped = golds_by_pull(mine)
    unplaceable = len(mine) - sum(len(v) for v in grouped.values())
    logger.info("%s pairs across %s pull requests", len(mine), len(grouped))

    outcomes: list[Outcome] = []
    per_repo_outcomes: dict[str, list[Outcome]] = defaultdict(list)
    ranks: Counter[int] = Counter()

    async with httpx.AsyncClient() as http:
        client = GitHubClient(http, mining_token() or "", HttpCache(DEFAULT_CACHE_DIR))
        await client.assert_mining_budget()
        for repo in require_tune(repos):
            pulls = sorted(n for (r, n) in grouped if r.lower() == repo.lower())
            if not pulls:
                continue
            index = load_index(store, repo, encoder)
            for number in pulls[:limit] if limit else pulls:
                entries = grouped[(repo, number)]
                golds = [gold for _, gold in entries]
                try:
                    found, placed = await measure_pull(client, index, repo, number, golds)
                except GitHubError as error:
                    logger.warning("%s#%s could not be read: %s", repo, number, error)
                    continue
                outcomes.extend(found)
                per_repo_outcomes[repo].extend(found)
                ranks.update(rank for rank in placed if rank is not None)
                logger.info("%s#%s: %s", repo, number, Counter(str(outcome) for outcome in found))

    return render(
        summarise(outcomes),
        {repo: summarise(found) for repo, found in per_repo_outcomes.items()},
        ranks,
        unplaceable,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-assembly", description=__doc__)
    parser.add_argument("--repos", nargs="*", default=None, help="Defaults to the tune split.")
    parser.add_argument(
        "--pulls-per-repo",
        type=int,
        default=0,
        help="Cap per repo, for a quick look. 0 measures every pull request that has a pair.",
    )
    parser.add_argument("--sink", choices=("local", "huggingface"), default="huggingface")
    parser.add_argument("--corpus-dir", default=".corpus")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    encoder = SentenceEncoder(MODELS["gte-modernbert"])
    markdown = asyncio.run(
        run(
            build_store(args.sink, args.corpus_dir),
            encoder,
            args.repos or tune_repos(),
            args.pulls_per_repo,
        )
    )
    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
