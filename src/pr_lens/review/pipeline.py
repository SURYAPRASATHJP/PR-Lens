"""One pull request in, the comments worth leaving out, or a stated reason for silence.

Shared by replay and by the live job, which differ only in what they fetch beforehand and
what they do with the result. Nothing here reads GitHub or writes anywhere.

Every draft the model writes is kept with its fate, including the ones that never survive,
because the keep-or-kill loop and Phase 8's dataset both need to see what was thrown away
and why. The gates run cheapest first, and every one of them can only remove a draft:

    self_critique     the model's own specific, non_obvious and grounded, all three true
    not_commentable   anchored to a line GitHub would refuse, or to a hunk never shown
    duplicate         a second draft on a line already drafted
    already_commented a human, or an earlier run, already commented on that line
    filtered          the second call, which sees the drafts and not the context, said no
    unjudged          still standing when the second call failed, so never posted
    over_cap          past MAX_COMMENTS_PER_PR, in the order the model wrote them

Silence has a reason every time, and a rate limit is one of them. It never becomes a
comment, because a failed call produces no drafts to post.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from pr_lens.ingest.diff import Hunk
from pr_lens.review import prompts
from pr_lens.review.context import Block, PullRequest, Skipped, commentable, fit, render_imports
from pr_lens.review.provider import (
    CALL_TOKEN_BUDGET,
    Completion,
    Inference,
    InferenceUnavailable,
    estimate_tokens,
)
from pr_lens.review.retrieve import CommentIndex, PastComment
from pr_lens.review.tools import SCHEMAS, Grounding, Toolbox
from pr_lens.review.verify import Verification
from pr_lens.review.verify import render as render_verification

logger = logging.getLogger(__name__)

# A reviewer nobody asked for who leaves ten comments reads as noise, and noise rate is
# the second headline number. Three is room for the one comment that matters plus a
# second opinion. The posted_comments ledger holds the same number as a constraint.
MAX_COMMENTS_PER_PR = 3

# Hunks considered at all, in rank order. Only the first few fit the budget, and each one
# considered costs a retrieval query and possibly a file fetch.
MAX_CANDIDATE_HUNKS = 8


def draft_frame(
    pull: "PullRequest",
    windows: Mapping[str, Sequence[str]],
    *,
    verification: "Verification | None" = None,
    tools: object = None,
) -> tuple[str, str, int]:
    """The system prompt, the head block, and the tokens left for hunks after both.

    One definition because two callers need it to agree. `review` assembles a prompt with
    it; `eval/assembly.py` measures which hunks that assembly keeps, and a budget computed
    twice is a budget that drifts, which would make the measurement quietly describe a
    pipeline that does not exist.
    """
    system = prompts.DRAFT_SYSTEM.format(cap=MAX_COMMENTS_PER_PR)
    if tools is not None:
        system = f"{system}\n\n{prompts.TOOL_SYSTEM}"
    # Only regressions render, so this is empty unless the sandbox found a test that passes
    # at the base commit and fails at this pull request's head.
    evidence = render_verification(verification) if verification else ""
    # The import block of every changed file, ahead of the hunks and outside their budget.
    scope = render_imports(windows)
    head = "\n\n".join(part for part in (prompts.header(pull), evidence, scope) if part)
    budget = (
        CALL_TOKEN_BUDGET
        - DRAFT_COMPLETION_TOKENS
        - estimate_tokens(system)
        - estimate_tokens(head)
        - (TOOL_RESERVE_TOKENS if tools is not None else 0)
    )
    return system, head, budget


# Completion room inside the per-call budget. gpt-oss reasons before it answers, and the
# reasoning counts against max_tokens, so the drafting call gets far more than its JSON
# needs. provider_check reports the real completion sizes.
DRAFT_COMPLETION_TOKENS = 2000
FILTER_COMPLETION_TOKENS = 1000

# Turns of tool calling before the answer is asked for, when a toolbox is given. A loop
# re-sends its whole conversation every turn, so cost grows with the square of this. Four
# is room to read a file, grep for a definition and check the history, which is what the
# eight kills in batch a needed, and not room to wander.
MAX_TOOL_TURNS = 4
TOOL_TURN_TOKENS = 700

# The research conversation must fit one call with room for that turn's answer. A tool
# turn's completion is TOOL_TURN_TOKENS, not the drafting call's, because since the
# research phase was split out they are two separate calls and the drafting call's budget
# has nothing to do with the size of the conversation here.
#
# Subtracting DRAFT_COMPLETION_TOKENS as well, which is what the split left behind, made
# the ceiling 4,300 while a full prompt is exactly 4,300: the hunk budget is
# CALL_TOKEN_BUDGET - DRAFT_COMPLETION_TOKENS - system - head - TOOL_RESERVE_TOKENS, so
# system plus user comes to the same number. Every real review tripped the ceiling before
# its first turn and recorded as a tools run that looked nothing up. Two pull requests on
# 2026-09-20-c-tools did exactly that before the batch was stopped.
TOOL_CONVERSATION_CEILING = CALL_TOKEN_BUDGET - TOOL_TURN_TOKENS

# Room kept back from the hunk budget for the findings block. The drafting call carries a
# summary of what the tools returned, not the conversation that produced it, so this is
# the size of the summary rather than of the loop.
TOOL_RESERVE_TOKENS = 700

# The findings block handed to the drafting call. One tool result is a window or a handful
# of matching lines, so this holds several and cuts the rest.
MAX_FINDINGS_CHARS = 2000


class NoComment(StrEnum):
    # The pull request could not be read: a deleted head commit, a vanished repository.
    FETCH_FAILED = "fetch_failed"
    NOTHING_REVIEWABLE = "nothing_reviewable"
    TOO_LARGE = "too_large"
    MODEL_SILENT = "model_silent"
    FILTERED_ALL = "filtered_all"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"


class Fate(StrEnum):
    KEPT = "kept"
    SELF_CRITIQUE = "self_critique"
    NOT_COMMENTABLE = "not_commentable"
    DUPLICATE = "duplicate"
    ALREADY_COMMENTED = "already_commented"
    FILTERED = "filtered"
    UNJUDGED = "unjudged"
    OVER_CAP = "over_cap"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DraftItem(_Strict):
    path: str
    line: int
    body: str
    evidence: str
    critique: str
    specific: bool
    non_obvious: bool
    grounded: bool


class DraftAnswer(_Strict):
    plan: str
    drafts: list[DraftItem]
    no_comment_reason: str


class Verdict(_Strict):
    index: int
    keep: bool
    reason: str


class FilterAnswer(_Strict):
    verdicts: list[Verdict]


@dataclass(frozen=True, slots=True)
class Draft:
    item: DraftItem
    fate: Fate
    filter_reason: str = ""


@dataclass(frozen=True, slots=True)
class Call:
    step: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str | None

    @classmethod
    def of(cls, step: str, completion: Completion) -> "Call":
        return cls(
            step,
            completion.provider,
            completion.model,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            completion.seconds,
            completion.finish_reason,
        )


@dataclass(frozen=True, slots=True)
class Review:
    no_comment: NoComment | None
    detail: str = ""
    plan: str = ""
    drafts: list[Draft] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    shown: list[Hunk] = field(default_factory=list)
    dropped: list[Hunk] = field(default_factory=list)
    # Ranked past MAX_CANDIDATE_HUNKS, so never costed against the budget at all. A big
    # pull request has hundreds of these and none of them is an assembly failure.
    unconsidered: list[Hunk] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    past_shown: int = 0
    verification: Verification | None = None
    grounding: Grounding = field(default_factory=Grounding)

    @property
    def kept(self) -> list[DraftItem]:
        return [draft.item for draft in self.drafts if draft.fate is Fate.KEPT]


def _parse[T: BaseModel](model: type[T], completion: Completion) -> T | None:
    if completion.finish_reason == "length":
        return None
    try:
        return model.model_validate(json.loads(completion.content))
    except (json.JSONDecodeError, ValidationError):
        return None


def _line_texts(hunks: Sequence[Hunk]) -> dict[tuple[str, int], str]:
    return {(hunk.path, n): text for hunk in hunks for n, text in hunk.new_file_lines()}


async def review(
    pull: PullRequest,
    hunks: Sequence[Hunk],
    skipped: Sequence[Skipped],
    windows: Mapping[str, Sequence[str]],
    inference: Inference,
    *,
    index: CommentIndex | None = None,
    past_before: int | None = None,
    existing: frozenset[tuple[str, int]] = frozenset(),
    verification: Verification | None = None,
    toolbox: Toolbox | None = None,
) -> Review:
    """Draft, gate and filter. `past_before` is the retrieval cutoff, the pull request's own
    number unless a seeded replay says otherwise; `existing` is every (path, line) already
    carrying a comment, which live review fills and replay leaves empty, since there the
    human's comments are the answer key."""
    skipped = list(skipped)
    candidates = list(hunks[:MAX_CANDIDATE_HUNKS])
    if not candidates:
        return Review(NoComment.NOTHING_REVIEWABLE, skipped=skipped, verification=verification)

    past: list[list[PastComment]] = [[] for _ in candidates]
    if index is not None:
        before = past_before if past_before is not None else pull.number
        past = index.search([hunk.text for hunk in candidates], before=before)

    system, head, budget = draft_frame(pull, windows, verification=verification, tools=toolbox)
    blocks = [
        Block(hunk, prompts.block_variants(n, hunk, windows.get(hunk.path), past[n - 1]))
        for n, hunk in enumerate(candidates, start=1)
    ]
    fitted = fit(blocks, budget)
    # Two different losses, kept apart. `dropped` is what the budget could not fit and is
    # the assembler's own number; `unconsidered` is what ranked past the candidate ceiling
    # and was never costed at all. Batch 2026-09-18-b averaged 18.6 "dropped" hunks and
    # one pull request reported 192, which is not a budget failure, it is MAX_CANDIDATE_HUNKS.
    # Added together they say the assembler is losing almost everything, which is false and
    # points at the wrong fix.
    unconsidered = list(hunks[MAX_CANDIDATE_HUNKS:])
    dropped = fitted.dropped
    shown = [hunk for hunk, _ in fitted.shown]
    if not shown:
        return Review(
            NoComment.TOO_LARGE,
            skipped=skipped,
            dropped=dropped,
            unconsidered=unconsidered,
            verification=verification,
        )
    # fit no longer keeps a prefix of the blocks, so each shown block carries its rank.
    past_shown = sum(
        len(past[position])
        for position, (_, text) in zip(fitted.positions, fitted.shown, strict=True)
        if prompts.PAST_HEADING in text
    )
    base = Review(
        None,
        shown=shown,
        dropped=dropped,
        unconsidered=unconsidered,
        skipped=skipped,
        past_shown=past_shown,
        verification=verification,
    )

    user = "\n\n".join([head, *(text for _, text in fitted.shown)])
    calls: list[Call] = []
    findings = ""
    try:
        if toolbox is not None:
            research = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            calls, findings = await _look_things_up(research, inference, toolbox)
            # Recorded before the drafting call, not after. The first version set it
            # afterwards, so the runs that failed were exactly the runs with no record of
            # what the loop had done, which is when it matters most.
            base = replace(base, grounding=Grounding(len(calls), tuple(toolbox.used)))
        # The drafting call is the ordinary one. It never carries the tool instructions or
        # the `tools` parameter, only what the research found, as text.
        drafted = await inference.complete(
            [
                {"role": "system", "content": prompts.DRAFT_SYSTEM.format(cap=MAX_COMMENTS_PER_PR)},
                {"role": "user", "content": "\n\n".join(part for part in (user, findings) if part)},
            ],
            max_tokens=DRAFT_COMPLETION_TOKENS,
            schema=prompts.DRAFT_SCHEMA,
        )
    except InferenceUnavailable as unavailable:
        reason = NoComment.RATE_LIMITED if unavailable.rate_limited else NoComment.UNAVAILABLE
        return replace(base, no_comment=reason, calls=calls, detail=str(unavailable))
    calls.append(Call.of("draft", drafted))
    answer = _parse(DraftAnswer, drafted)
    if answer is None:
        logger.warning("unparseable draft answer: %r", drafted.content[:300])
        return replace(
            base, no_comment=NoComment.MALFORMED, calls=calls, detail=drafted.content[:300]
        )
    base = replace(base, calls=calls, plan=answer.plan)
    if not answer.drafts:
        return replace(base, no_comment=NoComment.MODEL_SILENT, detail=answer.no_comment_reason)

    drafts, candidates_for_filter = _gate(answer.drafts, shown, existing)
    if not candidates_for_filter:
        return replace(base, no_comment=NoComment.FILTERED_ALL, drafts=drafts)

    code = _line_texts(shown)
    listing = [
        (item.path, item.line, code.get((item.path, item.line), ""), item.body)
        for _, item in candidates_for_filter
    ]
    try:
        filtered = await inference.complete(
            [
                {"role": "system", "content": prompts.FILTER_SYSTEM},
                {"role": "user", "content": prompts.filter_input(listing)},
            ],
            max_tokens=FILTER_COMPLETION_TOKENS,
            schema=prompts.FILTER_SCHEMA,
        )
    except InferenceUnavailable as unavailable:
        reason = NoComment.RATE_LIMITED if unavailable.rate_limited else NoComment.UNAVAILABLE
        return replace(base, no_comment=reason, drafts=drafts, detail=str(unavailable))
    calls.append(Call.of("filter", filtered))
    verdicts = _parse(FilterAnswer, filtered)
    if verdicts is None:
        logger.warning("unparseable filter answer: %r", filtered.content[:300])
        return replace(
            base,
            no_comment=NoComment.MALFORMED,
            drafts=drafts,
            calls=calls,
            detail=filtered.content[:300],
        )

    decided = {verdict.index: verdict for verdict in verdicts.verdicts}
    kept = 0
    for position, (slot, _) in enumerate(candidates_for_filter):
        verdict = decided.get(position)
        # A draft the filter did not mention is dropped: silence is the safe default.
        if verdict is None or not verdict.keep:
            why = verdict.reason if verdict else "no verdict given"
            drafts[slot] = replace(drafts[slot], fate=Fate.FILTERED, filter_reason=why)
        elif kept >= MAX_COMMENTS_PER_PR:
            drafts[slot] = replace(drafts[slot], fate=Fate.OVER_CAP, filter_reason=verdict.reason)
        else:
            drafts[slot] = replace(drafts[slot], fate=Fate.KEPT, filter_reason=verdict.reason)
            kept += 1

    return replace(
        base,
        no_comment=None if kept else NoComment.FILTERED_ALL,
        drafts=drafts,
        calls=calls,
    )


def _conversation_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """What the provider will count, near enough.

    Not `estimate_tokens(json.dumps(messages))`, which is what the first version did. JSON
    escapes every newline into two characters, so a conversation carrying a diff reads
    about forty percent larger than it is and the ceiling trips before the first turn.
    encode/httpx#3371 got zero turns that way and the run still recorded as a tools run.
    """
    return sum(
        estimate_tokens(str(message.get("content") or "")) for message in messages
    ) + 4 * len(messages)


async def _look_things_up(
    messages: Sequence[Mapping[str, Any]], inference: Inference, toolbox: Toolbox
) -> tuple[list[Call], str]:
    """A research phase, separate from drafting. Returns its calls and what it found.

    The conversation here is the model's own: assistant turns exactly as the provider sent
    them, then one tool message per call. It is NOT what the drafting call is then given.
    The first version replayed this whole conversation into the drafting call with the tool
    instructions still in the system prompt and no `tools` in the request, and Groq refused
    the lot: "Tool choice is none, but model called a tool", a 400 that loses the entire
    review. Three of seven pull requests died that way on 20 Sep before the batch was
    stopped. The model is not wrong to try; it was told it had tools and then offered none.

    So the loop's output is a findings block, plain text, and the drafting call is the
    ordinary one with that block added. Nothing downstream mentions a tool, so there is
    nothing for the model to reach for and nothing for the provider to reject. It is also
    cheaper, because the drafting call no longer re-sends the research.

    Stops on the first turn that calls nothing, on MAX_TOOL_TURNS, or when the conversation
    has grown past what one call can carry.
    """
    conversation = list(messages)
    calls: list[Call] = []
    findings: list[str] = []
    for turn in range(MAX_TOOL_TURNS):
        if _conversation_tokens(conversation) > TOOL_CONVERSATION_CEILING:
            logger.info("tool loop stopped at turn %s: no room left in the call", turn)
            break
        try:
            completion = await inference.complete(
                conversation, max_tokens=TOOL_TURN_TOKENS, tools=SCHEMAS
            )
        except InferenceUnavailable as unavailable:
            # Research is enrichment. Failing to enrich is not failing to review, so this
            # stops looking things up and drafts with whatever it already has. pypa/hatch#624
            # lost an entire review to a provider 400 over a misspelled tool argument.
            logger.info("tool loop stopped at turn %s: %s", turn, unavailable)
            break
        calls.append(Call.of(f"tools.{turn}", completion))
        if not completion.tool_calls:
            break
        conversation.append(completion.message)
        for asked in completion.tool_calls:
            answer = await toolbox.call(asked.name, asked.arguments)
            conversation.append({"role": "tool", "tool_call_id": asked.id, "content": answer})
            findings.append(answer)
    return calls, prompts.findings(findings, MAX_FINDINGS_CHARS)


def _gate(
    items: Sequence[DraftItem],
    shown: Sequence[Hunk],
    existing: frozenset[tuple[str, int]],
) -> tuple[list[Draft], list[tuple[int, DraftItem]]]:
    """The deterministic gates. Returns every draft with a provisional fate, and the ones
    still standing, by position, for the filter call to judge."""
    allowed: dict[str, frozenset[int]] = {}
    for hunk in shown:
        allowed[hunk.path] = allowed.get(hunk.path, frozenset()) | commentable(hunk)
    drafts: list[Draft] = []
    standing: list[tuple[int, DraftItem]] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        where = (item.path, item.line)
        if not (item.specific and item.non_obvious and item.grounded):
            fate = Fate.SELF_CRITIQUE
        elif item.line not in allowed.get(item.path, frozenset()):
            fate = Fate.NOT_COMMENTABLE
        elif where in seen:
            fate = Fate.DUPLICATE
        elif where in existing:
            fate = Fate.ALREADY_COMMENTED
        else:
            # Provisional. The filter call decides between kept, filtered and over cap.
            fate = Fate.UNJUDGED
            standing.append((len(drafts), item))
            seen.add(where)
        drafts.append(Draft(item, fate))
    return drafts, standing
