"""The keep-or-kill file: every draft of a replay batch, beside what the human said.

`export` writes one markdown file for a batch. The user writes keep or kill, and why if
they like, on each VERDICT line, and `import` reads those back into drafts.verdict. Every
draft is listed, the ones the gates threw away included, because a gate that is too strict
only shows up when someone reads what it discarded.

The file belongs in the workspace's notes/drafts/, outside the public repository, so
--out has no default. Everything the model wrote is quoted, so no line of a draft can be
mistaken for a verdict.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pr_lens.db import reviews
from pr_lens.db.connection import connect
from pr_lens.db.reviews import Verdict
from pr_lens.logging import configure

logger = logging.getLogger(__name__)

_VERDICT = re.compile(r"^VERDICT (?P<draft>\d+):[ \t]*(?P<text>.*)$")
_DECISION = re.compile(r"^(?P<word>keep|kill)\b[\s,:;.]*(?P<reason>.*)$", re.IGNORECASE)


class VerdictError(ValueError):
    """A VERDICT line that is neither blank, keep nor kill. Refused rather than guessed."""


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in (text.strip() or "(empty)").splitlines()]


def _reference(value: Any) -> list[dict[str, Any]]:
    # asyncpg hands jsonb back as text unless a codec is registered.
    parsed = json.loads(value) if isinstance(value, str) else value
    return list(parsed or [])


def render(
    batch: str,
    rows: Sequence[Mapping[str, Any]],
    silent: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = [
        f"# Replay batch `{batch}`",
        "",
        "On each VERDICT line write keep or kill, then why if you like. Leave it blank to",
        "skip. Would you leave this comment on a stranger's pull request under your name?",
        "",
    ]
    current: tuple[str, int] | None = None
    for row in rows:
        pull = (row["repo"], row["pr_number"])
        if pull != current:
            current = pull
            repo, number = pull
            lines += [
                f"## {repo}#{number}",
                "",
                f"https://github.com/{repo}/pull/{number} at `{str(row['head_sha'])[:7]}`",
                "",
                "The model's plan:",
                *_quote(row["plan"]),
                "",
                "What the human reviewers said on this pull request:",
            ]
            for said in _reference(row["reference"]):
                lines += _quote(f"{said['path']} ({said['author']}): {said['body']}")
                lines += [f"> {said['url']}", ""]
        verdict = row["verdict"] or ""
        reason = row["verdict_reason"] or ""
        lines += [
            f"### draft {row['draft_id']} | {row['fate']} | {row['path']}:{row['line']}",
            "",
            *_quote(row["body"]),
            "",
            "evidence:",
            *_quote(row["evidence"]),
            "critique:",
            *_quote(row["critique"]),
        ]
        if row["filter_reason"]:
            lines += ["filter:", *_quote(row["filter_reason"])]
        lines += ["", f"VERDICT {row['draft_id']}: {verdict} {reason}".rstrip(), ""]
    if silent:
        # Nothing to judge here, but a pull request the provider refused must not read the
        # same as one the model looked at and had nothing to say about.
        lines += ["## Drafted nothing", ""]
        for run in silent:
            detail = f": {run['detail']}" if run["detail"] else ""
            lines.append(f"- {run['repo']}#{run['pr_number']} {run['no_comment']}{detail}")
        lines.append("")
    return "\n".join(lines)


def parse(text: str) -> list[tuple[int, Verdict, str]]:
    """Every filled VERDICT line as (draft id, keep or kill, reason)."""
    verdicts = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _VERDICT.match(line)
        if match is None or not match["text"].strip():
            continue
        decision = _DECISION.match(match["text"].strip())
        if decision is None:
            raise VerdictError(f"line {number}: {line!r} is neither keep nor kill")
        word: Verdict = "keep" if decision["word"].lower() == "keep" else "kill"
        verdicts.append((int(match["draft"]), word, decision["reason"].strip()))
    return verdicts


async def batch_rows(
    dsn: str, batch: str
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """The batch's drafts, and the pull requests that produced none."""
    conn = await connect(dsn)
    try:
        return (
            list(await reviews.batch_drafts(conn, batch)),
            list(await reviews.silent_runs(conn, batch)),
        )
    finally:
        await conn.close()


async def save(dsn: str, verdicts: Sequence[tuple[int, Verdict, str]]) -> None:
    conn = await connect(dsn)
    try:
        await reviews.record_verdicts(conn, verdicts)
    finally:
        await conn.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-verdicts", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--batch", required=True)
    exporting.add_argument("--out", type=Path, required=True)
    importing = commands.add_parser("import")
    importing.add_argument("--file", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        logger.error("DATABASE_URL is not set")
        return 1
    if args.command == "export":
        rows, silent = asyncio.run(batch_rows(dsn, args.batch))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(args.batch, rows, silent), encoding="utf-8")
        logger.info("%s drafts and %s silent runs written to %s", len(rows), len(silent), args.out)
    else:
        verdicts = parse(args.file.read_text(encoding="utf-8"))
        asyncio.run(save(dsn, verdicts))
        logger.info("%s verdicts recorded", len(verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
