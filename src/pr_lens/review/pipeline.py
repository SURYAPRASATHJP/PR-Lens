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

from pydantic import BaseModel, ConfigDict, ValidationError

from pr_lens.ingest.diff import Hunk
from pr_lens.review import prompts
from pr_lens.review.context import Block, PullRequest, Skipped, commentable, fit
from pr_lens.review.provider import (
    CALL_TOKEN_BUDGET,
    Completion,
    Inference,
    InferenceUnavailable,
    estimate_tokens,
)
from pr_lens.review.retrieve import CommentIndex, PastComment

logger = logging.getLogger(__name__)

# A reviewer nobody asked for who leaves ten comments reads as noise, and noise rate is
# the second headline number. Three is room for the one comment that matters plus a
# second opinion. The posted_comments ledger holds the same number as a constraint.
MAX_COMMENTS_PER_PR = 3

# Hunks considered at all, in rank order. Only the first few fit the budget, and each one
# considered costs a retrieval query and possibly a file fetch.
MAX_CANDIDATE_HUNKS = 8

# Completion room inside the per-call budget. gpt-oss reasons before it answers, and the
# reasoning counts against max_tokens, so the drafting call gets far more than its JSON
# needs. provider_check reports the real completion sizes.
DRAFT_COMPLETION_TOKENS = 2000
FILTER_COMPLETION_TOKENS = 1000


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
    skipped: list[Skipped] = field(default_factory=list)
    past_shown: int = 0

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
) -> Review:
    """Draft, gate and filter. `past_before` is the retrieval cutoff, the pull request's own
    number unless a seeded replay says otherwise; `existing` is every (path, line) already
    carrying a comment, which live review fills and replay leaves empty, since there the
    human's comments are the answer key."""
    skipped = list(skipped)
    candidates = list(hunks[:MAX_CANDIDATE_HUNKS])
    if not candidates:
        return Review(NoComment.NOTHING_REVIEWABLE, skipped=skipped)

    past: list[list[PastComment]] = [[] for _ in candidates]
    if index is not None:
        before = past_before if past_before is not None else pull.number
        past = index.search([hunk.text for hunk in candidates], before=before)

    system = prompts.DRAFT_SYSTEM.format(cap=MAX_COMMENTS_PER_PR)
    head = prompts.header(pull)
    budget = (
        CALL_TOKEN_BUDGET
        - DRAFT_COMPLETION_TOKENS
        - estimate_tokens(system)
        - estimate_tokens(head)
    )
    blocks = [
        Block(hunk, prompts.block_variants(n, hunk, windows.get(hunk.path), past[n - 1]))
        for n, hunk in enumerate(candidates, start=1)
    ]
    fitted = fit(blocks, budget)
    dropped = fitted.dropped + list(hunks[MAX_CANDIDATE_HUNKS:])
    shown = [hunk for hunk, _ in fitted.shown]
    if not shown:
        return Review(NoComment.TOO_LARGE, skipped=skipped, dropped=dropped)
    # fit keeps a prefix of the blocks, so the n-th shown block is the n-th candidate.
    past_shown = sum(
        len(past[n]) for n, (_, text) in enumerate(fitted.shown) if prompts.PAST_HEADING in text
    )
    base = Review(None, shown=shown, dropped=dropped, skipped=skipped, past_shown=past_shown)

    user = "\n\n".join([head, *(text for _, text in fitted.shown)])
    try:
        drafted = await inference.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=DRAFT_COMPLETION_TOKENS,
            schema=prompts.DRAFT_SCHEMA,
        )
    except InferenceUnavailable as unavailable:
        reason = NoComment.RATE_LIMITED if unavailable.rate_limited else NoComment.UNAVAILABLE
        return replace(base, no_comment=reason, detail=str(unavailable))
    calls = [Call.of("draft", drafted)]
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
