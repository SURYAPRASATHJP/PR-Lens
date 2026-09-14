"""Raw test output in, structured failures out, bounded either way.

Phase 4 hands this to a drafting model, so two properties matter more than coverage.
It never raises on output it does not recognise: an unfamiliar reporter, a truncated
stream or plain garbage degrades to a bounded excerpt of the tail, which is where every
runner puts its summary. And it is capped in record count and in size, so a suite with
four thousand failures yields a readable answer and a count of what was left out, not an
unbounded blob.

Parsers are regular expressions over the formats the supported runners print, written
against output captured from real runs in the sandbox images, which lives in
tests/fixtures/sandbox. The detected framework's parser is tried first and the others
after it, because scripts.test often runs something other than what package.json's
dependencies suggest.
"""

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from pr_lens.sandbox.detect import Framework
from pr_lens.sandbox.spec import WORK_DIR

MAX_FAILURES = 25
MAX_MESSAGE_CHARS = 400
MAX_NAME_CHARS = 300
EXCERPT_CHARS = 4000

# Glyphs the runners print that look like ASCII and are not, named so nobody "fixes" them.
_JEST_SEPARATOR = "\u203a"  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
_VITEST_POINTER = "\u276f"  # HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT
_NODE_INFO = "\u2139"  # INFORMATION SOURCE

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True, slots=True)
class Failure:
    test: str
    kind: str
    file: str | None
    line: int | None
    message: str


@dataclass(frozen=True, slots=True)
class Counts:
    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None


@dataclass(frozen=True, slots=True)
class Evidence:
    """parser names the format that matched, or None when nothing did and excerpt is all
    there is. omitted counts failures beyond MAX_FAILURES."""

    parser: str | None
    failures: tuple[Failure, ...]
    counts: Counts | None
    omitted: int = 0
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Parsed:
    failures: list[Failure]
    counts: Counts | None

    @property
    def empty(self) -> bool:
        return not self.failures and self.counts is None


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _relative(path: str) -> str:
    prefix = WORK_DIR + "/"
    path = path.removeprefix(prefix)
    return path.removeprefix("./")


def _failure(test: str, kind: str, file: str | None, line: int | None, message: str) -> Failure:
    return Failure(
        test=_clip(test, MAX_NAME_CHARS),
        kind=kind,
        file=_relative(file) if file else None,
        line=line,
        message=_clip(message, MAX_MESSAGE_CHARS),
    )


def _first(preferred: list[re.Match[str]], rest: list[re.Match[str]]) -> re.Match[str] | None:
    if preferred:
        return preferred[0]
    return rest[0] if rest else None


def _count(summary: str, *words: str) -> int | None:
    for word in words:
        match = re.search(rf"(\d+) {word}\b", summary)
        if match:
            return int(match[1])
    return None


# pytest

_PYTEST_SUMMARY_ENTRY = re.compile(r"^(FAILED|ERROR) (.+?)(?: - (.*))?$")
_PYTEST_BLOCK = re.compile(r"^_{3,} (.+?) _{3,}$")
_PYTEST_FINAL = re.compile(r"^=+ (.*\d+ (?:passed|failed|errors?|skipped).*) in [\d.]+s", re.M)
# pytest-pretty replaces the final line with a block, one count to a line. Found on
# pydantic-settings in the real-data run, 14 Sep 2026.
_PYTEST_PRETTY = re.compile(r"^Results \([\d.]+s\):\n((?:[ \t]+\d+ \w+\n?)+)", re.M)
_PYTEST_LOCATION = re.compile(r"^([^\s:]+\.py):(\d+): ")


def _pytest_blocks(lines: list[str]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        header = _PYTEST_BLOCK.match(line)
        if header:
            current = blocks.setdefault(header[1], [])
        elif line.startswith("=") and current is not None:
            current = None
        elif current is not None:
            current.append(line)
    return blocks


def _pytest_block_for(nodeid: str, kind: str, blocks: dict[str, list[str]]) -> list[str]:
    path, _, rest = nodeid.partition("::")
    if kind == "error":
        if not rest:
            return blocks.get(f"ERROR collecting {path}", [])
        name = rest.split("::")[-1]
        for phase in ("setup", "teardown"):
            if f"ERROR at {phase} of {name}" in blocks:
                return blocks[f"ERROR at {phase} of {name}"]
        return []
    return blocks.get(rest.replace("::", "."), [])


def _parse_pytest(text: str) -> _Parsed:
    lines = text.splitlines()
    blocks = _pytest_blocks(lines)
    failures: list[Failure] = []
    in_summary = False
    for line in lines:
        if "short test summary info" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        entry = _PYTEST_SUMMARY_ENTRY.match(line)
        if not entry:
            continue
        kind = "failed" if entry[1] == "FAILED" else "error"
        nodeid = entry[2]
        block = _pytest_block_for(nodeid, kind, blocks)
        path = nodeid.partition("::")[0]
        location = [m for m in map(_PYTEST_LOCATION.match, block) if m and m[1] == path]
        errors = [line[4:] for line in block if line.startswith("E   ")]
        message = entry[3] or (errors[-1] if errors else "")
        failures.append(
            _failure(nodeid, kind, path, int(location[-1][2]) if location else None, message)
        )

    final = _PYTEST_FINAL.findall(text) or _PYTEST_PRETTY.findall(text)
    counts = None
    if final:
        summary = final[-1]
        counts = Counts(
            passed=_count(summary, "passed"),
            failed=_count(summary, "failed"),
            errors=_count(summary, "errors", "error"),
            skipped=_count(summary, "skipped"),
        )
    return _Parsed(failures, counts)


# jest

_JEST_BLOCK = re.compile(r"^  ● (.+)$")
_JEST_SUITE = re.compile(r"^(?:FAIL|PASS) (\S+)")
_JEST_FRAME = re.compile(r"^\s+at (?:.*? \()?([^()\s]+?):(\d+):\d+\)?$")
_JEST_TOTALS = re.compile(r"^Tests:\s+(.+)$", re.M)


def _parse_jest(text: str) -> _Parsed:
    failures: list[Failure] = []
    suite: str | None = None
    title: str | None = None
    body: list[str] = []

    def close() -> None:
        if title is None:
            return
        message = next((line.strip() for line in body if line.strip()), "")
        frames = [m for m in map(_JEST_FRAME.match, body) if m]
        frames = [m for m in frames if "node_modules" not in m[1] and not m[1].startswith("node:")]
        suite_path = _relative(suite) if suite else None
        frame = _first([m for m in frames if _relative(m[1]) == suite_path], frames)
        kind = "error" if title == "Test suite failed to run" else "failed"
        failures.append(
            _failure(
                title.replace(f" {_JEST_SEPARATOR} ", " > "),
                kind,
                frame[1] if frame else suite,
                int(frame[2]) if frame else None,
                message,
            )
        )

    for line in text.splitlines():
        suite_match = _JEST_SUITE.match(line)
        block = _JEST_BLOCK.match(line)
        if suite_match or block or line.startswith("Test Suites:"):
            close()
            title, body = None, []
            if suite_match:
                suite = suite_match[1]
            if block:
                title = block[1]
            continue
        if title is not None:
            body.append(line)
    close()

    totals = _JEST_TOTALS.findall(text)
    counts = None
    if totals:
        counts = Counts(
            passed=_count(totals[-1], "passed"),
            failed=_count(totals[-1], "failed"),
            skipped=_count(totals[-1], "skipped"),
        )
    return _Parsed(failures, counts)


# vitest

_VITEST_BLOCK = re.compile(r"^ FAIL  (\S+)(?: > (.+)| \[ .+ \])?$")
_VITEST_FRAME = re.compile(rf"^ {_VITEST_POINTER} (?:\S+ )?([^\s:]+):(\d+):\d+$")
_VITEST_TOTALS = re.compile(r"^\s+Tests\s+(.+)$", re.M)


def _parse_vitest(text: str) -> _Parsed:
    failures: list[Failure] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        block = _VITEST_BLOCK.match(line)
        if not block:
            continue
        path, name = block[1], block[2]
        body: list[str] = []
        for following in lines[index + 1 :]:
            if _VITEST_BLOCK.match(following) or following.startswith("⎯"):
                break
            body.append(following)
        message = next((entry.strip() for entry in body if entry.strip()), "")
        frames = [m for m in map(_VITEST_FRAME.match, body) if m]
        frame = _first([m for m in frames if m[1] == path], frames)
        failures.append(
            _failure(
                name or path,
                "failed" if name else "error",
                frame[1] if frame else path,
                int(frame[2]) if frame else None,
                message,
            )
        )

    totals = _VITEST_TOTALS.findall(text)
    counts = None
    if totals:
        counts = Counts(
            passed=_count(totals[-1], "passed"),
            failed=_count(totals[-1], "failed"),
            skipped=_count(totals[-1], "skipped"),
        )
    return _Parsed(failures, counts)


# node:test, spec reporter and TAP

_NODE_SPEC_LOCATION = re.compile(r"^test at (.+):(\d+):\d+$")
_NODE_SPEC_NAME = re.compile(r"^✖ (.+?)(?: \([\d.]+m?s\))?$")
_NODE_TAP_FAIL = re.compile(r"^(\s*)not ok \d+ - (.+)$")
_NODE_TAP_LOCATION = re.compile(r"^\s+location: '(.+):(\d+):\d+'$")
_NODE_TAP_ERROR = re.compile(r"^\s+error: (.*)$")
_SUBTESTS_FAILED = re.compile(r"^'?\d+ subtests? failed'?$")


def _node_counts(text: str, marker: str) -> Counts | None:
    def number(word: str) -> int | None:
        match = re.search(rf"^{re.escape(marker)} {word} (\d+)$", text, re.M)
        return int(match[1]) if match else None

    counts = Counts(passed=number("pass"), failed=number("fail"), skipped=number("skipped"))
    return None if counts == Counts() else counts


def _parse_node_spec(text: str) -> _Parsed:
    failures: list[Failure] = []
    lines = text.splitlines()
    if "✖ failing tests:" in lines:
        lines = lines[lines.index("✖ failing tests:") + 1 :]
        for index, line in enumerate(lines):
            location = _NODE_SPEC_LOCATION.match(line)
            if not location or index + 1 >= len(lines):
                continue
            name = _NODE_SPEC_NAME.match(lines[index + 1])
            if not name:
                continue
            message = next(
                (entry.strip() for entry in lines[index + 2 :] if entry.strip()),
                "",
            )
            failures.append(_failure(name[1], "failed", location[1], int(location[2]), message))
    return _Parsed(failures, _node_counts(text, _NODE_INFO))


def _parse_node_tap(text: str) -> _Parsed:
    failures: list[Failure] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        failed = _NODE_TAP_FAIL.match(line)
        if not failed:
            continue
        file = number = None
        message = ""
        for following in lines[index + 1 :]:
            if following.strip() == "...":
                break
            location = _NODE_TAP_LOCATION.match(following)
            if location:
                file, number = location[1], int(location[2])
            error = _NODE_TAP_ERROR.match(following)
            if error:
                message = error[1]
                if message in ("|-", "|", ">-"):
                    position = lines.index(following, index)
                    rest = (entry.strip() for entry in lines[position + 1 :])
                    message = next((entry for entry in rest if entry), "")
        if _SUBTESTS_FAILED.match(message):
            # A describe block reporting that its children failed. The children are listed.
            continue
        failures.append(_failure(failed[2], "failed", file, number, message.strip("'")))
    return _Parsed(failures, _node_counts(text, "#"))


_PARSERS: dict[str, Callable[[str], _Parsed]] = {
    "pytest": _parse_pytest,
    "jest": _parse_jest,
    "vitest": _parse_vitest,
    "node-test-spec": _parse_node_spec,
    "node-test-tap": _parse_node_tap,
}

_FIRST: dict[Framework, tuple[str, ...]] = {
    Framework.PYTEST: ("pytest",),
    Framework.JEST: ("jest",),
    Framework.VITEST: ("vitest",),
    Framework.NODE_TEST: ("node-test-spec", "node-test-tap"),
    Framework.UNKNOWN: (),
}


def _excerpt(text: str) -> str:
    text = text.strip()
    return text if len(text) <= EXCERPT_CHARS else "..." + text[-(EXCERPT_CHARS - 3) :]


def parse(framework: Framework, stdout: str, stderr: str) -> Evidence:
    """Never raises. What matched, or a bounded excerpt of the tail when nothing did.

    Both streams are read as one, stdout first: jest reports on stderr, vitest splits its
    report across the two, and pytest uses stdout alone.
    """
    text = f"{stdout}\n{stderr}" if stderr.strip() else stdout
    # Plugins built on rich colour their output even under --color=no.
    text = _ANSI.sub("", text)
    order = [*_FIRST[framework], *(name for name in _PARSERS if name not in _FIRST[framework])]
    for name in order:
        parsed = _PARSERS[name](text)
        if parsed.empty:
            continue
        reported_failures = (parsed.counts.failed or 0) if parsed.counts else 0
        excerpt = _excerpt(text) if reported_failures and not parsed.failures else ""
        return Evidence(
            parser=name,
            failures=tuple(parsed.failures[:MAX_FAILURES]),
            counts=parsed.counts,
            omitted=max(0, len(parsed.failures) - MAX_FAILURES),
            excerpt=excerpt,
        )
    return Evidence(parser=None, failures=(), counts=None, excerpt=_excerpt(text))
