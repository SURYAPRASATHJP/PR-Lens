"""The Actions entrypoint reports every ending as a row. Docker and GitHub are faked here;
the real thing runs in sandbox.yml."""

import io
import json
import tarfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from pr_lens.jobs import sandbox as job
from pr_lens.sandbox import runner, session
from pr_lens.sandbox.spec import FetchSpec, Outcome, SandboxResult, SandboxSpec

PYTEST_OUTPUT = (
    Path(__file__).parent / "fixtures" / "sandbox" / "pytest-failures.stdout"
).read_text(encoding="utf-8")


def result(**overrides: object) -> SandboxResult:
    values: dict[str, object] = {
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 2.5,
    }
    values.update(overrides)
    return SandboxResult(**values)  # type: ignore[arg-type]


def tarball(files: dict[str, str], top: str = "owner-repo-abc123") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(f"{top}/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../../../etc/passwd"
        archive.addfile(link)
    return buffer.getvalue()


def test_extract_strips_the_top_directory_and_refuses_links_out_of_the_tree(
    tmp_path: Path,
) -> None:
    skipped = session.extract(tarball({"pyproject.toml": "x", "tests/test_a.py": ""}), tmp_path)
    assert (tmp_path / "pyproject.toml").read_text() == "x"
    assert (tmp_path / "tests" / "test_a.py").exists()
    assert not (tmp_path / "escape").exists()
    assert skipped == 1


class FakeSandbox:
    """Stands in for runner.pull, fetch, run and deps_volume, recording what it was given."""

    def __init__(self, fetched: SandboxResult, ran: SandboxResult) -> None:
        self.fetched = fetched
        self.ran = ran
        self.specs: list[FetchSpec | SandboxSpec] = []

    async def pull(self, image: str) -> float:
        return 4.25

    async def fetch(self, spec: FetchSpec) -> SandboxResult:
        self.specs.append(spec)
        return self.fetched

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        self.specs.append(spec)
        return self.ran

    @asynccontextmanager
    async def deps_volume(self) -> AsyncIterator[str]:
        yield "pr-lens-fake"


@pytest.fixture
def python_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_source(
        repo: str, into: Path, sha: str | None = None, token: str | None = None
    ) -> tuple[str, float, int]:
        (into / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '1'\n")
        (into / "tests").mkdir()
        (into / "tests" / "test_a.py").write_text("")
        return "a" * 40, 1.5, 0

    monkeypatch.setattr(session, "source_at", fake_source)


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeSandbox) -> None:
    for name in ("pull", "fetch", "run", "deps_volume"):
        monkeypatch.setattr(runner, name, getattr(fake, name))


@pytest.mark.usefixtures("python_repo")
@pytest.mark.parametrize(
    ("ran", "outcome"),
    [
        (result(exit_code=1, stdout=PYTEST_OUTPUT), Outcome.FAILED),
        (result(exit_code=None, timed_out=True, docker_exit_code=137), Outcome.TIMED_OUT),
        (result(exit_code=1, oom_killed=True, memory_peak_bytes=2**31), Outcome.OOM_KILLED),
        (result(exit_code=5), Outcome.NO_TESTS),
        (result(exit_code=None, docker_exit_code=125, stderr="docker: error"), Outcome.ERROR),
        (result(exit_code=0), Outcome.PASSED),
    ],
    ids=["failed", "timeout", "oom", "collected-nothing", "error", "passed"],
)
async def test_every_ending_is_a_row(
    monkeypatch: pytest.MonkeyPatch, ran: SandboxResult, outcome: Outcome
) -> None:
    fake = FakeSandbox(fetched=result(), ran=ran)
    install(monkeypatch, fake)
    row = await session.run_repo("encode/httpx")
    assert row.outcome == outcome.value
    assert row.pull_seconds == 4.25
    assert row.sha == "a" * 40
    assert row.fetch_outcome == "passed"


@pytest.mark.usefixtures("python_repo")
async def test_a_failed_run_carries_parsed_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeSandbox(result(), result(exit_code=1, stdout=PYTEST_OUTPUT)))
    row = await session.run_repo("encode/httpx")
    assert row.evidence is not None
    assert row.evidence["parser"] == "pytest"
    assert len(row.evidence["failures"]) == 4


@pytest.mark.usefixtures("python_repo")
async def test_a_failed_install_is_reported_and_the_tests_are_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSandbox(
        fetched=result(exit_code=1, stderr="ERROR: No matching distribution found for x"),
        ran=result(),
    )
    install(monkeypatch, fake)
    row = await session.run_repo("encode/httpx")
    assert row.outcome == Outcome.INSTALL_FAILED.value
    assert "No matching distribution" in row.excerpt
    assert [type(spec) for spec in fake.specs] == [FetchSpec]


async def test_a_repository_with_no_tests_never_touches_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_source(
        repo: str, into: Path, sha: str | None = None, token: str | None = None
    ) -> tuple[str, float, int]:
        (into / "go.mod").write_text("module x\n")
        return "b" * 40, 0.1, 0

    async def forbidden(*args: object) -> float:
        raise AssertionError("docker was called for a repository with no tests")

    monkeypatch.setattr(session, "source_at", fake_source)
    monkeypatch.setattr(runner, "pull", forbidden)
    row = await session.run_repo("encode/httpx")
    assert row.outcome == Outcome.NO_TESTS.value
    assert "Go" in row.reason


@pytest.mark.usefixtures("python_repo")
async def test_a_docker_failure_is_an_error_row_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken_pull(image: str) -> float:
        raise runner.SandboxError("docker pull failed: denied")

    monkeypatch.setattr(runner, "pull", broken_pull)
    row = await session.run_repo("encode/httpx")
    assert row.outcome == Outcome.ERROR.value
    assert row.error == "docker pull failed: denied"


def test_the_matrix_is_the_mining_set_and_fits_a_workflow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert job.main(["sandbox", "repos"]) == 0
    repos = json.loads(capsys.readouterr().out)
    assert len(repos) == 27
    assert 0 < len(repos) <= 256


def test_run_refuses_a_repository_outside_the_mining_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_REPO", "someone/else")
    assert job.main(["sandbox", "run"]) == 1


def test_an_annotation_survives_newlines_and_percent_signs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    job.annotate("sandbox-row a/b", 'line one\nline two 100%\r{"k": 1}')
    out = capsys.readouterr().out
    assert out == '::notice title=sandbox-row a/b 1/1::line one%0Aline two 100%25%0D{"k": 1}\n'


def test_a_long_row_is_split_under_githubs_annotation_cap_and_reassembles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """GitHub cuts an annotation at 4096 bytes. Eleven of the first 27 rows were cut."""
    row = json.dumps({"excerpt": "x" * 10_000, "repo": "a/b"})
    job.annotate("sandbox-row a/b", row)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    chunks = []
    for index, line in enumerate(lines, start=1):
        header, _, body = line.removeprefix("::").partition("::")
        assert header == f"notice title=sandbox-row a/b {index}/3"
        assert len(body.encode()) < 4096
        chunks.append(body)
    assert json.loads("".join(chunks)) == json.loads(row)


def test_the_table_renders_from_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = tmp_path / "rows"
    rows.mkdir()
    for name, outcome, pull in (("a/one", "passed", 3.0), ("b/two", "no_tests", None)):
        row = session.Row(repo=name, outcome=outcome, pull_seconds=pull, reason="why | not")
        (rows / f"{name.replace('/', '__')}.json").write_text(json.dumps(asdict(row)))
    table = tmp_path / "table.md"
    monkeypatch.setenv("SANDBOX_ROWS", str(rows))
    monkeypatch.setenv("SANDBOX_TABLE", str(table))
    monkeypatch.setenv("SANDBOX_SOURCE", "sandbox.yml run 1 at abc")
    assert job.main(["sandbox", "table"]) == 0
    text = table.read_text()
    assert "Rows from sandbox.yml run 1 at abc." in text
    assert "| passed | 1 |" in text
    assert "| no_tests | 1 |" in text
    assert "image pull   p50 3.0s" in text
    assert "why / not" in text
