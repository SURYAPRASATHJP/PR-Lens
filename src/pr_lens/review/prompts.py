"""The two calls' instructions, their output schemas, and how their inputs are laid out.

Two calls, not PLAN.md's four, because the free tier's 8K tokens a minute is smaller than
one call carrying the diff and its context (notes/phase-4-provider-check.md). The first
call plans, drafts and critiques in one structured answer; the second sees only the drafts
and the line each is anchored to, never the retrieval context, and decides what survives.
Self-critique inside one call is weaker than a separate pass. That is the cost the tier
imposes, and the keep-or-kill loop is what says whether it is affordable.

The schemas are strict: every property required, nothing extra, which is what a provider's
structured-output mode needs. pipeline.py validates the answer against pydantic models with
the same fields, and a test holds the two in step.
"""

from collections.abc import Sequence
from typing import Any

from pr_lens.ingest.diff import Hunk
from pr_lens.review.context import PullRequest, render
from pr_lens.review.retrieve import PastComment

# The body of a pull request is the author's framing, worth a paragraph and no more.
MAX_BODY_CHARS = 600

DRAFT_SYSTEM = """\
You review one pull request as a careful senior engineer who knows this repository. Leave \
a comment only where a maintainer would thank you for it: a bug, an edge case that breaks, \
a contract the change violates, a missing check, a regression the tests would not catch. \
Say nothing about style, naming, formatting, docstrings, typing nits, or anything a linter \
would flag. Do not praise, summarise, or ask for tests in general.

Silence is a correct answer. Most pull requests deserve no comment. Never draft more than \
{cap}.

The code listing has one line per row: a line number, a marker, then the code. Markers: \
"+" added, " " unchanged but inside the diff, "-" removed (it has no number), "." \
surrounding code outside the diff. You may only comment on a numbered row marked "+" or " \
", and should prefer "+". Past review comments from this repository may follow a hunk, \
labelled like [2.1]. They show what this project's reviewers care about. Treat them as \
hints: never copy one, and never comment only because a past comment exists.

Before the hunks you are shown the import lines of each changed file at this commit. That \
list is complete for the top of those files, so if a name appears there it IS imported and \
in scope. Never write that a name is undefined, unimported or missing when it is in that \
list. You are shown a few hunks of the change, not the whole file and not the whole \
repository, so a function, class or name you cannot see usually exists. If a comment \
depends on code you were not shown, do not write it.

Answer with JSON only:
- plan: under 60 words, what could break in this change.
- drafts: each comment you would leave, with path; line (a number from the listing); body \
(under 80 words, concrete, says what breaks and when, no greeting); evidence (the line or \
past comment it rests on); critique (the strongest reason this comment is wrong, obvious, \
or not worth the author's time); and three honest booleans. specific: it is about this \
code, not a general rule. non_obvious: the author probably missed it. grounded: it \
follows from what is shown, not from guessing about code you cannot see.
- no_comment_reason: when drafts is empty, why; otherwise an empty string."""

FILTER_SYSTEM = """\
You are the last check before a comment is posted, under a bot's name, on a pull request \
written by someone who did not ask for your opinion. Drop most drafts. Keep a draft only \
if every one of these holds: it is about the quoted line, not the code in general; it \
names a concrete failure or risk, not a preference; the author would not already know \
it; it is not a question fishing for information; it is short and polite. Answer with \
JSON only: one verdict per draft, by its index, with keep and a one-sentence reason."""

DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "drafts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "body": {"type": "string"},
                    "evidence": {"type": "string"},
                    "critique": {"type": "string"},
                    "specific": {"type": "boolean"},
                    "non_obvious": {"type": "boolean"},
                    "grounded": {"type": "boolean"},
                },
                "required": [
                    "path",
                    "line",
                    "body",
                    "evidence",
                    "critique",
                    "specific",
                    "non_obvious",
                    "grounded",
                ],
                "additionalProperties": False,
            },
        },
        "no_comment_reason": {"type": "string"},
    },
    "required": ["plan", "drafts", "no_comment_reason"],
    "additionalProperties": False,
}

FILTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "keep", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def header(pull: PullRequest) -> str:
    body = pull.body.strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + " [cut]"
    lines = [f"Pull request: {pull.title.strip()}"]
    if body:
        lines += ["", body]
    return "\n".join(lines)


PAST_HEADING = "Past review comments in this repository, nearest first:"


def _past(position: int, comments: Sequence[PastComment]) -> str:
    rows = [PAST_HEADING]
    for n, comment in enumerate(comments, start=1):
        where = f"{comment.path} " if comment.path else ""
        rows.append(f"[{position}.{n}] {where}(#{comment.pull_request_number}) {comment.body}")
    return "\n".join(rows)


def block_variants(
    position: int,
    hunk: Hunk,
    file_lines: Sequence[str] | None,
    comments: Sequence[PastComment],
) -> tuple[str, ...]:
    """Richest first: window and past comments, then without the window, then the bare hunk."""
    title = f"Hunk {position}"
    bare = f"{title}\n{render(hunk)}"
    variants = []
    if comments:
        if file_lines is not None:
            variants.append(f"{title}\n{render(hunk, file_lines)}\n{_past(position, comments)}")
        variants.append(f"{bare}\n{_past(position, comments)}")
    elif file_lines is not None:
        variants.append(f"{title}\n{render(hunk, file_lines)}")
    variants.append(bare)
    return tuple(variants)


def filter_input(drafts: Sequence[tuple[str, int, str, str]]) -> str:
    """Each draft as (path, line, the code on that line, body), numbered from zero."""
    rows = []
    for index, (path, line, code, body) in enumerate(drafts):
        rows += [f"Draft {index}", f"{path}:{line}: {code.strip()}", body, ""]
    return "\n".join(rows).rstrip()
