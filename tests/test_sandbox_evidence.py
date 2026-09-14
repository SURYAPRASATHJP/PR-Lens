"""Evidence parsed from output captured from real runs, 14 Sep 2026.

The fixtures in tests/fixtures/sandbox are what pytest 9.1.1, jest, vitest 5 and node 22's
test runner actually printed for small failing suites, run in python:3.12-slim and
node:22-slim with the checkout at /work. None of it is written by hand.
"""

import random
from pathlib import Path

import pytest

from pr_lens.sandbox.detect import Framework
from pr_lens.sandbox.evidence import (
    EXCERPT_CHARS,
    MAX_FAILURES,
    MAX_MESSAGE_CHARS,
    Counts,
    Failure,
    parse,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sandbox"


def captured(name: str) -> tuple[str, str]:
    return (
        (FIXTURES / f"{name}.stdout").read_text(encoding="utf-8"),
        (FIXTURES / f"{name}.stderr").read_text(encoding="utf-8"),
    )


def test_pytest_failures_and_setup_errors_become_located_records() -> None:
    evidence = parse(Framework.PYTEST, *captured("pytest-failures"))
    assert evidence.parser == "pytest"
    assert evidence.counts == Counts(passed=1, failed=3, errors=1, skipped=None)
    assert evidence.failures == (
        Failure(
            "tests/test_math.py::test_wrong", "failed", "tests/test_math.py", 13, "assert 4 == 5"
        ),
        Failure(
            "tests/test_math.py::test_raises",
            "failed",
            "tests/test_math.py",
            17,
            "ValueError: bad input: 42",
        ),
        Failure(
            "tests/test_math.py::TestThing::test_method",
            "failed",
            "tests/test_math.py",
            31,
            "assert [1, 2] == [1, 3]",
        ),
        Failure(
            "tests/test_math.py::test_uses_broken",
            "error",
            "tests/test_math.py",
            22,
            "RuntimeError: fixture exploded",
        ),
    )
    assert evidence.excerpt == ""


def test_a_pytest_collection_error_carries_the_import_that_failed() -> None:
    """The common real-world case: the suite never ran, because a module would not import."""
    evidence = parse(Framework.PYTEST, *captured("pytest-collection-error"))
    assert evidence.counts is not None and evidence.counts.errors == 1
    assert evidence.failures == (
        Failure(
            "tests/test_import.py",
            "error",
            "tests/test_import.py",
            1,
            "ModuleNotFoundError: No module named 'not_a_real_module'",
        ),
    )


def test_pytest_pretty_counts_are_read_through_its_colour_codes() -> None:
    """pydantic-settings in the real-data run: pytest-pretty's summary block, bold in ANSI
    even with --color=no, parsed as nothing at all until this fixture existed."""
    evidence = parse(Framework.PYTEST, *captured("pytest-pretty"))
    assert evidence.parser == "pytest"
    assert evidence.counts == Counts(passed=787, skipped=5)


def test_jest_reports_on_stderr_and_locates_each_failure_in_the_test_file() -> None:
    evidence = parse(Framework.JEST, *captured("jest"))
    assert evidence.parser == "jest"
    assert evidence.counts == Counts(passed=1, failed=3)
    assert evidence.failures == (
        Failure(
            "sum > adds negative numbers",
            "failed",
            "sum.test.js",
            8,
            "expect(received).toBe(expected) // Object.is equality",
        ),
        Failure(
            "parse rejects garbage",
            "failed",
            "sum.test.js",
            12,
            "SyntaxError: Expected property name or '}' in JSON at position 1 (line 1 column 2)",
        ),
        Failure(
            "throws on purpose", "failed", "sum.test.js", 15, "TypeError: config.port is undefined"
        ),
    )


def test_vitest_splits_its_report_across_both_streams() -> None:
    evidence = parse(Framework.VITEST, *captured("vitest"))
    assert evidence.parser == "vitest"
    assert evidence.counts == Counts(passed=1, failed=3)
    assert [(f.test, f.file, f.line) for f in evidence.failures] == [
        ("sum > adds negative numbers", "sum.test.js", 8),
        ("parse rejects garbage", "sum.test.js", 12),
        ("throws on purpose", "sum.test.js", 15),
    ]
    assert (
        evidence.failures[0].message == "AssertionError: expected -3 to be -4 // Object.is equality"
    )


def test_node_test_spec_reporter() -> None:
    evidence = parse(Framework.NODE_TEST, *captured("node-test-spec"))
    assert evidence.parser == "node-test-spec"
    assert evidence.counts == Counts(passed=1, failed=2, skipped=0)
    assert evidence.failures == (
        Failure(
            "adds negative numbers",
            "failed",
            "sum.test.js",
            5,
            "AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:",
        ),
        Failure(
            "throws on purpose", "failed", "sum.test.js", 6, "TypeError: config.port is undefined"
        ),
    )


def test_node_test_tap_reporter_which_node_22_uses_when_not_on_a_terminal() -> None:
    evidence = parse(Framework.NODE_TEST, *captured("node-test-tap"))
    assert evidence.parser == "node-test-tap"
    assert evidence.counts == Counts(passed=1, failed=2, skipped=0)
    assert [(f.test, f.file, f.line) for f in evidence.failures] == [
        ("adds negative numbers", "sum.test.js", 5),
        ("throws on purpose", "sum.test.js", 6),
    ]
    assert evidence.failures[0].message == "Expected values to be strictly equal:"
    assert evidence.failures[1].message == "config.port is undefined"


def test_an_unknown_framework_still_finds_the_format_by_trying_each_parser() -> None:
    assert parse(Framework.UNKNOWN, *captured("jest")).parser == "jest"
    assert parse(Framework.JEST, *captured("vitest")).parser == "vitest"


def test_unrecognised_output_degrades_to_a_bounded_excerpt_of_the_tail() -> None:
    noise = "".join(f"line {number} of something no parser knows\n" for number in range(5000))
    evidence = parse(Framework.PYTEST, noise, "")
    assert evidence.parser is None
    assert evidence.failures == ()
    assert len(evidence.excerpt) <= EXCERPT_CHARS
    assert evidence.excerpt.endswith("line 4999 of something no parser knows")


def test_a_summary_with_failures_but_no_records_keeps_the_excerpt() -> None:
    evidence = parse(Framework.PYTEST, "=== 2 failed, 1 passed in 0.1s ===\n", "")
    assert evidence.counts is not None and evidence.counts.failed == 2
    assert evidence.failures == ()
    assert "2 failed" in evidence.excerpt


def test_thousands_of_failures_are_capped_and_counted() -> None:
    lines = ["=========================== short test summary info ============================"]
    lines += [f"FAILED tests/test_x.py::test_{n} - assert {'x' * 2000}" for n in range(4000)]
    lines.append("======================= 4000 failed in 9.0s =======================")
    evidence = parse(Framework.PYTEST, "\n".join(lines), "")
    assert len(evidence.failures) == MAX_FAILURES
    assert evidence.omitted == 4000 - MAX_FAILURES
    assert all(len(f.message) <= MAX_MESSAGE_CHARS for f in evidence.failures)


@pytest.mark.parametrize("name", ["pytest-failures", "jest", "vitest", "node-test-tap"])
def test_truncated_output_never_raises(name: str) -> None:
    stdout, stderr = captured(name)
    text = stdout + stderr
    for cut in range(0, len(text), 37):
        parse(Framework.UNKNOWN, text[:cut], "")
        parse(Framework.UNKNOWN, text[cut:], "")


def test_random_garbage_never_raises() -> None:
    generator = random.Random(14092026)  # noqa: S311 -- a fixed seed, not a secret
    alphabet = "abc FAILED ERROR ● ✖ \u276f \u203a not ok :: - _=\n0123456789.py'|"
    for _ in range(300):
        text = "".join(generator.choice(alphabet) for _ in range(generator.randint(0, 600)))
        evidence = parse(Framework.UNKNOWN, text, text[::-1])
        assert len(evidence.excerpt) <= EXCERPT_CHARS


def test_evidence_serialises_to_plain_json_types() -> None:
    evidence = parse(Framework.PYTEST, *captured("pytest-failures"))
    as_dict = evidence.to_dict()
    assert as_dict["parser"] == "pytest"
    assert as_dict["failures"][0]["line"] == 13
