from dataclasses import fields
from pathlib import Path

import pytest

from pr_lens.sandbox.spec import (
    MEMORY_LIMIT_MB,
    TMP_TMPFS_MB,
    WORK_TMPFS_MB,
    Fetcher,
    FetchSpec,
    Outcome,
    SandboxResult,
    SandboxSpec,
    SpecError,
    check_requirement,
)

IMAGE = "ghcr.io/example/sandbox@sha256:" + "a" * 64


def spec(**overrides: object) -> SandboxSpec:
    values: dict[str, object] = {
        "image": IMAGE,
        "command": ("python", "-m", "pytest"),
        "source_dir": Path("checkout"),
    }
    values.update(overrides)
    return SandboxSpec(**values)  # type: ignore[arg-type]


def test_the_spec_has_exactly_these_fields_and_none_of_them_is_a_network_switch() -> None:
    """Adding a field to SandboxSpec fails here first, which is the point. A new field is a
    new way to configure isolation, and that deserves a deliberate edit to this list."""
    assert {field.name for field in fields(SandboxSpec)} == {
        "image",
        "command",
        "source_dir",
        "setup",
        "deps_volume",
        "env",
        "timeout_seconds",
    }


def test_the_fetch_step_has_no_command_to_set() -> None:
    names = {field.name for field in fields(FetchSpec)}
    assert "command" not in names
    assert "setup" not in names
    assert "env" not in names


def test_the_writable_trees_fit_inside_the_memory_cap_with_room_to_run() -> None:
    """tmpfs is charged to the memory cgroup. A tmpfs bigger than the cap gets the whole
    container OOM-killed before it ever reports ENOSPC, measured 14 Sep 2026."""
    assert WORK_TMPFS_MB + TMP_TMPFS_MB <= MEMORY_LIMIT_MB // 2


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/example/sandbox:latest",
        "ghcr.io/example/sandbox",
        "ghcr.io/example/sandbox@sha256:abc",
        "ghcr.io/example/sandbox:1.0@sha256:" + "a" * 64,
        "ghcr.io/example/sandbox@sha256:" + "A" * 64,
    ],
)
def test_an_image_not_pinned_by_digest_is_refused(image: str) -> None:
    with pytest.raises(SpecError):
        spec(image=image)


def test_an_empty_command_and_a_non_positive_timeout_are_refused() -> None:
    with pytest.raises(SpecError):
        spec(command=())
    with pytest.raises(SpecError):
        spec(timeout_seconds=0)


def test_an_environment_name_docker_would_read_as_something_else_is_refused() -> None:
    with pytest.raises(SpecError):
        spec(env=(("A=B", "c"),))
    with pytest.raises(SpecError):
        spec(env=(("--privileged", "1"),))


def test_a_volume_this_sandbox_did_not_name_is_refused() -> None:
    """Otherwise a spec could mount any volume on the runner, docker's own included."""
    with pytest.raises(SpecError):
        spec(deps_volume="/var/run/docker.sock")
    assert spec(deps_volume="pr-lens-0123abcd").deps_volume == "pr-lens-0123abcd"


@pytest.mark.parametrize(
    "line",
    [
        "--index-url=https://evil.example/simple",
        "-e .",
        "-r other.txt",
        " requests",
        "pkg @ https://evil.example/pkg-1.0-py3-none-any.whl",
        "./local/path",
        "",
        "not a requirement!",
    ],
)
def test_a_requirement_pip_could_read_as_an_option_path_or_url_is_refused(line: str) -> None:
    with pytest.raises(SpecError):
        check_requirement(line)


@pytest.mark.parametrize(
    "line",
    ["requests", "httpx>=0.28", "attrs[tests]==24.2", 'tomli>=1; python_version < "3.11"'],
)
def test_an_ordinary_index_requirement_is_accepted(line: str) -> None:
    assert check_requirement(line).name


def test_requirements_belong_to_pip_alone() -> None:
    with pytest.raises(SpecError):
        FetchSpec(
            image=IMAGE,
            fetcher=Fetcher.NPM_CI,
            source_dir=Path("checkout"),
            deps_volume="pr-lens-abc",
            requirements=("left-pad",),
        )


def test_outcome_reads_timeouts_and_oom_before_the_exit_code() -> None:
    def result(**overrides: object) -> SandboxResult:
        values: dict[str, object] = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 1.0,
        }
        values.update(overrides)
        return SandboxResult(**values)  # type: ignore[arg-type]

    assert result().outcome is Outcome.PASSED
    assert result(exit_code=1).outcome is Outcome.FAILED
    assert result(exit_code=1, oom_killed=True).outcome is Outcome.OOM_KILLED
    assert result(exit_code=None, timed_out=True).outcome is Outcome.TIMED_OUT
    assert result(exit_code=None).outcome is Outcome.ERROR
