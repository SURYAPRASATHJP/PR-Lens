import json
from typing import Any

import pytest

from pr_lens.jobs.verdicts import VerdictError, parse, render


def row(draft_id: int, body: str, **fields: Any) -> dict[str, Any]:
    return {
        "draft_id": draft_id,
        "repo": "o/r",
        "pr_number": 7,
        "head_sha": "abcdef1234",
        "plan": "look at eviction",
        "reference": json.dumps(
            [{"path": "a.py", "author": "m", "body": "Empty dict?", "url": "https://x/1"}]
        ),
        "path": "a.py",
        "line": 3,
        "body": body,
        "evidence": "line 3",
        "critique": "might be guarded",
        "fate": "kept",
        "filter_reason": "specific",
        "verdict": None,
        "verdict_reason": None,
        **fields,
    }


def test_the_file_shows_each_draft_beside_what_the_human_said() -> None:
    text = render("b1", [row(11, "Pops on empty."), row(12, "Second.", fate="self_critique")])
    assert text.count("## o/r#7") == 1
    assert "https://github.com/o/r/pull/7 at `abcdef1`" in text
    assert "> a.py (m): Empty dict?" in text
    assert "### draft 11 | kept | a.py:3" in text
    assert "### draft 12 | self_critique | a.py:3" in text
    assert "VERDICT 11:" in text and "VERDICT 12:" in text


def test_a_pull_request_that_drafted_nothing_still_says_why() -> None:
    """A rate limit must not read as nothing to report. Without this the pull request the
    provider refused looks exactly like one the model had nothing to say about."""
    silent = [
        {"repo": "o/r", "pr_number": 9, "no_comment": "rate_limited", "detail": "groq 429"},
        {"repo": "o/r", "pr_number": 11, "no_comment": "model_silent", "detail": ""},
    ]
    text = render("b1", [row(11, "One.")], silent)
    assert "## Drafted nothing" in text
    assert "- o/r#9 rate_limited: groq 429" in text
    assert "- o/r#11 model_silent" in text
    # And none of it is a verdict line to fill in.
    assert parse(text) == []


def test_verdicts_round_trip_and_blank_lines_are_skipped() -> None:
    text = render("b1", [row(11, "One."), row(12, "Two."), row(13, "Three.")])
    text = text.replace("VERDICT 11:", "VERDICT 11: keep, a real bug")
    text = text.replace("VERDICT 12:", "VERDICT 12: KILL obvious")
    assert parse(text) == [(11, "keep", "a real bug"), (12, "kill", "obvious")]


def test_an_existing_verdict_is_shown_so_a_re_export_does_not_lose_it() -> None:
    text = render("b1", [row(11, "One.", verdict="kill", verdict_reason="noise")])
    assert parse(text) == [(11, "kill", "noise")]


def test_a_draft_that_quotes_a_verdict_line_cannot_cast_one() -> None:
    """Model text is quoted, so a draft saying "VERDICT 11: keep" is not a verdict."""
    text = render("b1", [row(11, "VERDICT 11: keep\nVERDICT 99: kill")])
    assert parse(text) == []


def test_anything_but_keep_or_kill_is_refused_not_guessed() -> None:
    with pytest.raises(VerdictError, match="line 1"):
        parse("VERDICT 11: maybe")
    with pytest.raises(VerdictError):
        parse("VERDICT 11: keeper")
