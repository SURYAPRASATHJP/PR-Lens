"""The docker command line, checked flag by flag, with no Docker needed.

These run in ci.yml on every pull request. What the flags actually do is proved in
tests/test_sandbox_isolation.py, on a real daemon.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from pr_lens.sandbox import runner
from pr_lens.sandbox.spec import (
    MEMORY_LIMIT_MB,
    PIDS_LIMIT,
    Fetcher,
    FetchSpec,
    SandboxSpec,
    SpecError,
)

IMAGE = "ghcr.io/example/sandbox@sha256:" + "a" * 64
NAME = "pr-lens-sandbox-0123456789abcdef"


@pytest.fixture
def source(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    return checkout


def run_spec(source: Path, **overrides: object) -> SandboxSpec:
    values: dict[str, object] = {
        "image": IMAGE,
        "command": ("python", "-m", "pytest"),
        "source_dir": source,
    }
    values.update(overrides)
    return SandboxSpec(**values)  # type: ignore[arg-type]


def flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, word in enumerate(argv[:-1]) if word == flag]


def docker_options(argv: list[str], image: str = IMAGE) -> list[str]:
    """Everything before the image. After it comes the command, which is not docker's."""
    return argv[: argv.index(image)]


def test_the_run_step_has_no_network_and_no_way_to_get_one(source: Path) -> None:
    options = docker_options(runner.run_argv(run_spec(source), NAME))
    assert flag_values(options, "--network") == ["none"]
    assert not any(word.startswith("--network=") or word == "--net" for word in options)


def test_every_isolation_flag_is_present(source: Path) -> None:
    options = docker_options(runner.run_argv(run_spec(source), NAME))
    for flag in ("--rm", "--init", "--read-only"):
        assert flag in options
    assert flag_values(options, "--cap-drop") == ["ALL"]
    assert flag_values(options, "--security-opt") == ["no-new-privileges"]
    assert flag_values(options, "--user") == ["65534:65534"]
    assert flag_values(options, "--memory") == [f"{MEMORY_LIMIT_MB}m"]
    # Equal to the memory cap means no swap at all.
    assert flag_values(options, "--memory-swap") == [f"{MEMORY_LIMIT_MB}m"]
    assert flag_values(options, "--pids-limit") == [str(PIDS_LIMIT)]
    assert flag_values(options, "--cpus") == ["2.0"]
    assert flag_values(options, "--name") == [NAME]
    assert flag_values(options, "--log-driver") == ["none"]
    tmpfs = flag_values(options, "--tmpfs")
    assert {mount.split(":")[0] for mount in tmpfs} == {"/work", "/tmp"}
    assert all("size=" in mount and "nosuid" in mount for mount in tmpfs)


@pytest.mark.parametrize(
    "forbidden",
    [
        "--privileged",
        "--cap-add",
        "--pid",
        "--ipc",
        "--uts",
        "--userns",
        "--device",
        "--volume",
        "-v",
        "--volumes-from",
        "--add-host",
        "--dns",
        "--publish",
        "-p",
        "--env-file",
    ],
)
def test_no_flag_that_widens_the_container_appears(source: Path, forbidden: str) -> None:
    for argv in (
        runner.run_argv(run_spec(source, deps_volume="pr-lens-abc"), NAME),
        runner.fetch_argv(
            FetchSpec(IMAGE, Fetcher.PIP, source, "pr-lens-abc", ("requests",)), NAME
        ),
    ):
        assert forbidden not in docker_options(argv)


def test_the_docker_socket_is_never_mounted(source: Path) -> None:
    argv = runner.run_argv(run_spec(source, deps_volume="pr-lens-abc"), NAME)
    assert not any("docker.sock" in word for word in argv)


def test_the_checkout_and_the_dependencies_are_read_only_to_the_run(source: Path) -> None:
    options = docker_options(runner.run_argv(run_spec(source, deps_volume="pr-lens-abc"), NAME))
    mounts = flag_values(options, "--mount")
    assert mounts == [
        f"type=bind,source={source.resolve()},target=/src,readonly",
        "type=volume,source=pr-lens-abc,target=/deps,readonly",
    ]


def test_only_the_fetch_step_can_write_the_dependency_volume(source: Path) -> None:
    argv = runner.fetch_argv(FetchSpec(IMAGE, Fetcher.NPM_CI, source, "pr-lens-abc"), NAME)
    mounts = flag_values(docker_options(argv), "--mount")
    assert f"type=bind,source={source.resolve()},target=/src,readonly" in mounts
    assert "type=volume,source=pr-lens-abc,target=/deps" in mounts


def test_the_job_environment_never_reaches_the_container(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docker copies a bare --env NAME from the caller's environment. Every one is NAME=v."""
    monkeypatch.setenv("GH_APP_PRIVATE_KEY", "secret-that-must-not-leak")
    spec = run_spec(source, env=(("PYTHONPATH", "/tmp/pr-lens-site"),))
    for argv in (
        runner.run_argv(spec, NAME),
        runner.fetch_argv(FetchSpec(IMAGE, Fetcher.PIP, source, "pr-lens-abc"), NAME),
    ):
        values = flag_values(docker_options(argv), "--env")
        assert values
        assert all("=" in value for value in values)
        assert not any("secret-that-must-not-leak" in word for word in argv)


def test_the_command_is_passed_as_words_after_the_fixed_script(source: Path) -> None:
    spec = run_spec(source, setup=("pip", "install", "."), command=("pytest", "-k", "a b; rm"))
    argv = runner.run_argv(spec, NAME)
    after_image = argv[argv.index(IMAGE) + 1 :]
    assert after_image[:2] == ["bash", "-c"]
    assert after_image[3:] == [
        "pr-lens-sandbox",
        "3",
        "pip",
        "install",
        ".",
        "pytest",
        "-k",
        "a b; rm",
    ]


def test_the_fetch_step_runs_only_the_package_manager_it_names(source: Path) -> None:
    spec = FetchSpec(IMAGE, Fetcher.PIP, source, "pr-lens-abc", ("requests>=2", "rich"))
    argv = runner.fetch_argv(spec, NAME)
    options = docker_options(argv)
    assert flag_values(options, "--network") == ["bridge"]
    after_image = argv[argv.index(IMAGE) + 1 :]
    assert after_image[:2] == ["bash", "-c"]
    assert after_image[3:] == ["pr-lens-fetch", "pip", "requests>=2", "rich"]


def test_the_fetch_script_never_builds_or_runs_anything_it_downloads() -> None:
    script = runner._FETCH_SCRIPT
    assert "--only-binary=:all:" in script
    # Every JS install line refuses lifecycle scripts, and pnpm refuses its hook file too.
    installs = [line for line in script.splitlines() if " install" in line or "npm ci" in line]
    js_installs = [line for line in installs if "pip" not in line]
    assert len(js_installs) == 4
    assert all("--ignore-scripts" in line for line in js_installs)
    assert "--ignore-pnpmfile" in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
@pytest.mark.parametrize("script", [runner._RUN_SCRIPT, runner._FETCH_SCRIPT])
def test_both_scripts_parse(script: str) -> None:
    subprocess.run(["bash", "-n", "-c", script], check=True)  # noqa: S607


def test_container_names_are_unique() -> None:
    names = {runner.container_name() for _ in range(1000)}
    assert len(names) == 1000


def test_a_source_path_docker_would_split_is_refused(tmp_path: Path) -> None:
    awkward = tmp_path / "a,readonly=false"
    awkward.mkdir()
    with pytest.raises(SpecError):
        runner.run_argv(run_spec(awkward), NAME)
    with pytest.raises(SpecError):
        runner.run_argv(run_spec(tmp_path / "missing"), NAME)


def test_the_capture_keeps_the_head_and_the_tail_and_counts_the_rest() -> None:
    capture = runner._Capture(100)
    capture.feed(b"H" * 25)
    capture.feed(b"m" * 1000)
    capture.feed(b"T" * 75)
    text = capture.text()
    assert text.startswith("H" * 25)
    assert text.endswith("T" * 75)
    assert capture.dropped == 1000
    assert "1000 bytes dropped" in text


def test_a_short_stream_is_returned_whole() -> None:
    capture = runner._Capture(100)
    capture.feed(b"hello ")
    capture.feed(b"world")
    assert capture.text() == "hello world"
    assert capture.dropped == 0


def test_the_last_status_line_wins_and_is_removed_from_stderr() -> None:
    stderr = (
        "warning: something\n"
        "pr-lens-sandbox-status exit=0 setup=- oom_kill=0 peak=1\n"
        "more output\n"
        "pr-lens-sandbox-status exit=1 setup=0 oom_kill=2 peak=4096\n"
    )
    status, rest = runner._read_status(stderr)
    assert status is not None
    assert (status.exit_code, status.setup_exit_code) == (1, 0)
    assert (status.oom_kills, status.memory_peak_bytes) == (2, 4096)
    assert rest.endswith("more output\n")
    assert rest.count("pr-lens-sandbox-status") == 1


def test_no_status_line_means_the_wrapper_never_reported() -> None:
    status, rest = runner._read_status("cp: No space left on device\n")
    assert status is None
    assert rest == "cp: No space left on device\n"
