"""The recall table as markdown, rendered from the result file and nothing else.

No number in the output is typed by hand. Everything below is read from the JSON the
recall job wrote, so the committed table and the stored result cannot drift apart.
"""

from collections.abc import Sequence
from typing import Any

from pr_lens.eval.recall import KS, interval

VARIANT_NAMES = {
    "before": "before-state (context and - lines)",
    "reviewed": "hunk as reviewed (the diff_hunk)",
}
SERIALISATION_NAMES = {"with_hunk": "comments indexed with hunk", "body_only": "comments body only"}


def render(result: dict[str, Any]) -> str:
    sections = [
        _header(result),
        _first_stage(result),
        _leakage(result),
        _truncation(result),
        _plateau(result),
        _post_filter(result),
        _latency(result),
        _query_set(result),
    ]
    return "\n\n".join(sections) + "\n"


def _header(result: dict[str, Any]) -> str:
    corpus = result["corpus"]
    models = ", ".join(
        f"`{m['name']}` at `{m['revision'][:10]}`, window {m['window']} tokens"
        for m in result["models"].values()
    )
    return "\n".join(
        [
            f"# Phase 2 recall table, query set {result['version']}",
            "",
            f"Query set sha256 `{result['pairs_sha256'][:16]}`, {result['query_set']['pairs']} "
            f"pairs from {corpus['repos']} tune repos. Index: {corpus['documents']:,} units "
            f"({_kinds(corpus['by_kind'])}). Exact search throughout, no ANN.",
            "",
            f"Models: {models}. Reranker `{result['reranker']['name']}` at "
            f"`{result['reranker']['revision'][:10]}`.",
            "",
            "Each query is a hunk a human reviewed; its one gold document is the review "
            "comment they left on it. recall@k is the share of queries whose gold is in the "
            "top k. The interval on recall@10 is a 95% Wald half-width.",
        ]
    )


def _first_stage(result: dict[str, Any]) -> str:
    lines = [
        "## 1. First stage, comments indexed body only",
        "",
        "The body-only rows are the honest numbers. Section 2 shows why.",
        "",
        _table_head(["query", "retriever", *[f"@{k}" for k in KS], "±@10"]),
    ]
    for variant in VARIANT_NAMES:
        for row in _rows(result, "body_only", variant):
            lines.append(_recall_line(VARIANT_NAMES[variant], _retriever(row), row))
    return "\n".join(lines)


def _leakage(result: dict[str, Any]) -> str:
    lines = [
        "## 2. Leakage: the same queries against the comment indexed with its hunk",
        "",
        "A review comment unit carries the diff_hunk it was left on, so querying with that "
        "hunk finds the gold by string match. The gap is the leakage, measured.",
        "",
        _table_head(["query", "retriever", "with hunk @10", "body only @10", "gap"]),
    ]
    for variant in VARIANT_NAMES:
        body = {_retriever(r): r for r in _rows(result, "body_only", variant)}
        for row in _rows(result, "with_hunk", variant):
            name = _retriever(row)
            with_hunk, body_only = row["recall"]["10"], body[name]["recall"]["10"]
            lines.append(
                f"| {VARIANT_NAMES[variant]} | {name} | {with_hunk:.3f} | {body_only:.3f} "
                f"| {with_hunk - body_only:+.3f} |"
            )
    return "\n".join(lines)


def _truncation(result: dict[str, Any]) -> str:
    lines = [
        "## 3. The truncation price",
        "",
        _table_head(
            ["model", "window", "units over 512", "units over window", "median unit tokens"]
        ),
    ]
    for key, stats in result["truncation"].items():
        units = stats["units"]
        lines.append(
            f"| {key} | {stats['window']} | {units['over_512']:.1%} | {units['over_window']:.1%} "
            f"| {units['median_tokens']:.0f} |"
        )
    lines += ["", "Units over 512 tokens, by kind, counted with each model's tokenizer:", ""]
    kinds = sorted(
        {k for s in result["truncation"].values() for k in s["units"]["over_512_by_kind"]}
    )
    lines.append(_table_head(["model", *kinds]))
    for key, stats in result["truncation"].items():
        by_kind = stats["units"]["over_512_by_kind"]
        lines.append(f"| {key} | " + " | ".join(f"{by_kind.get(k, 0.0):.1%}" for k in kinds) + " |")
    lines += [
        "",
        "Dense recall@10, body-only index, split by whether the query fits in 512 tokens. "
        "The gold is a comment body and rarely long, so this is where a 512 window pays:",
        "",
        _table_head(["model", "query", "fits in 512: n, @10", "longer: n, @10"]),
    ]
    for key, variants in result["buckets"].items():
        for variant, stats in variants.items():
            cells = [f"{n}, {r:.3f}" for n, r in stats["dense"].values()]
            lines.append(f"| {key} | {VARIANT_NAMES[variant]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plateau(result: dict[str, Any]) -> str:
    plateau = result.get("plateau")
    if not plateau:
        return "## 4. Reranker plateau\n\nNot computed yet: the rerank shards have not run."
    lines = [
        "## 4. Reranker plateau",
        "",
        f"The cross-encoder reorders the first N of the RRF list ({result['rerank']['model']} "
        f"plus BM25, body-only index), on a fixed sample of "
        f"{len(result['rerank']['items']) // 2} pairs. First stage @N is the ceiling: "
        "reranking can reorder the candidates, never add one.",
    ]
    for variant, rows in plateau.items():
        lines += [
            "",
            f"Query: {VARIANT_NAMES[variant]}",
            "",
            _table_head(["N", "first stage @N", "reranked @1", "reranked @5", "reranked @10"]),
        ]
        for row in rows:
            reranked = row["reranked"]
            cells = [f"{reranked[str(k)]:.3f}" if str(k) in reranked else "" for k in (1, 5, 10)]
            lines.append(
                f"| {row['depth']} | {row['first_stage']:.3f} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def _post_filter(result: dict[str, Any]) -> str:
    model = result["rerank"]["model"]
    lines = [
        "## 5. Post-filter recall collapse",
        "",
        f"Dense {model}, body-only index, recall@10 for a filtered question. Pre-filter "
        "searches only the rows the filter allows, exactly, and is the right answer. "
        "Post-filter fetches C candidates from the whole index and drops what the filter "
        "rejects, which is what pgvector's HNSW does with a WHERE clause. C = 40 is its "
        "default hnsw.ef_search. The gold always passes the filter, so every loss is the "
        "candidate budget running out.",
    ]
    for variant, filters in result["post_filter"][model].items():
        for name, stats in filters.items():
            lines += [
                "",
                f"{name}, keeping {stats['selectivity']:.2%} of the index on average. "
                f"Query: {VARIANT_NAMES[variant]}",
                "",
                _table_head(["search", "recall@10", "mean results", "queries short of 10"]),
            ]
            for row in stats["rows"]:
                label = (
                    "pre-filter, exact"
                    if row["candidates"] is None
                    else f"post-filter, C = {row['candidates']}"
                )
                lines.append(
                    f"| {label} | {row['recall']:.3f} | {row['mean_returned']:.1f} "
                    f"| {row['short_share']:.1%} |"
                )
    return "\n".join(lines)


def _latency(result: dict[str, Any]) -> str:
    machine = result["machine"]
    lines = [
        "## 6. Latency",
        "",
        f"Measured in the job that built this table: {machine['platform']}, "
        f"{machine['cpus']} CPUs. Each call timed alone, as serving would pay it.",
        "",
        _table_head(["step", "p50 ms", "p95 ms", "n"]),
    ]
    for name, stats in result["latency"].items():
        lines.append(f"| {name} | {stats['p50_ms']:.1f} | {stats['p95_ms']:.1f} | {stats['n']} |")
    neon = result["neon"]
    if "skipped" in neon:
        lines += ["", f"Neon: not measured, {neon['skipped']}."]
    else:
        warm = neon["resolve_30"]
        lines += [
            "",
            f"Neon, first query of the run including connect: "
            f"{neon['first_query_including_connect_ms']:.0f} ms. Resolving one list of 30 "
            f"candidates in one query afterwards: p50 {warm['p50_ms']:.1f} ms, p95 "
            f"{warm['p95_ms']:.1f} ms over {warm['n']}.",
        ]
    return "\n".join(lines)


def _query_set(result: dict[str, Any]) -> str:
    query_set = result["query_set"]
    lines = [
        "## 7. The query set",
        "",
        f"{query_set['pairs']} pairs, capped at {query_set['per_repo_cap']} per repo and 3 per "
        f"pull request. {query_set['with_suggestion_block']} carry a suggestion block, whose "
        "code overlaps the hunk lexically.",
        "",
        "Comments dropped, by the first rule each one failed:",
        "",
        _table_head(["rule", "comments"]),
        *(f"| {rule} | {count} |" for rule, count in query_set["dropped"].items()),
        "",
        _table_head(["repo", "comments read", "pairs", "RRF body-only recall@10, reviewed"]),
    ]
    per_repo = result["buckets"][result["rerank"]["model"]]["reviewed"]["per_repo_recall_at_10"]
    for repo, stats in sorted(query_set["repos"].items(), key=lambda kv: kv[0].lower()):
        recall = per_repo.get(repo)
        cell = f"{recall:.3f}" if recall is not None else ""
        lines.append(f"| {repo} | {stats['comments_seen']} | {stats['pairs']} | {cell} |")
    return "\n".join(lines)


def _rows(result: dict[str, Any], serialisation: str, variant: str) -> list[dict[str, Any]]:
    order = {"bm25": 0, "dense": 1, "rrf": 2}
    rows = [
        r
        for r in result["first_stage"]
        if r["serialisation"] == serialisation and r["variant"] == variant
    ]
    return sorted(rows, key=lambda r: (order[r["retriever"]], r["model"]))


def _retriever(row: dict[str, Any]) -> str:
    if row["retriever"] == "bm25":
        return "BM25"
    if row["retriever"] == "dense":
        return f"dense {row['model']}"
    return f"RRF {row['model']} + BM25"


def _recall_line(query: str, retriever: str, row: dict[str, Any]) -> str:
    recall = row["recall"]
    cells = " | ".join(f"{recall[str(k)]:.3f}" for k in KS)
    return f"| {query} | {retriever} | {cells} | {interval(recall['10'], row['n']):.3f} |"


def _table_head(columns: Sequence[str]) -> str:
    return "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|"


def _kinds(by_kind: dict[str, int]) -> str:
    return ", ".join(f"{count:,} {kind}" for kind, count in sorted(by_kind.items()))
