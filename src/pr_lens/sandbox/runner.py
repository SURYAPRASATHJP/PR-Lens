"""Turn a spec into a docker command line, run it under a wall clock, and read what happened.

Argv only, never a shell string built from input. The repository name, its file names and
its requirement strings reach this code from the public internet. The two bash scripts
below are constants: whatever varies is passed to them as positional arguments, which
bash never re-parses.

The read-only mount, decided 14 Sep 2026. PLAN.md says the checkout is mounted read-only,
and real suites write: pytest writes .pytest_cache and __pycache__, a build writes
egg-info beside the source, and many suites write scratch files next to their fixtures. A
literally read-only working tree fails a large share of real repositories for reasons
that have nothing to do with their tests. So the checkout is bind-mounted read-only at
/src, where nothing ever writes, and the run copies it into /work, a tmpfs, and works from
there. The root filesystem is read-only too, and the fetched dependencies are mounted
read-only. What the rule protects, that the code under review cannot change the checkout
it was handed, holds: tests/test_sandbox_isolation.py writes to /src, to / and to /deps
and expects EROFS for each. The run still gets the writable tree it needs, and its size
is charged against the memory cap because tmpfs lives in memory.

How a run ended is read from inside the container. The wrapper script prints one status
line after the command exits, carrying the exit code, the cgroup's oom_kill counter and
its peak memory. That is how an OOM kill of a test process is reported as OOM rather
than as a test failure: the test runner sees a dead worker and exits 1, and only the
cgroup knows why. When the whole container is killed, the wrapper cannot report, and a
docker exit of 137 that the wall clock did not cause can only be the kernel.
"""

import asyncio
import logging
import re
import time
import uuid
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from pr_lens.sandbox.spec import (
    CPU_LIMIT,
    DEPS_MOUNT,
    MEMORY_LIMIT_MB,
    OUTPUT_CAP_BYTES,
    PIDS_LIMIT,
    RUN_AS_USER,
    SOURCE_MOUNT,
    TMP_TMPFS_MB,
    WORK_DIR,
    WORK_TMPFS_MB,
    FetchSpec,
    Outcome,
    SandboxResult,
    SandboxSpec,
    SpecError,
)

logger = logging.getLogger(__name__)

DOCKER = "docker"

# Every container and volume this module creates carries it, so a leak is one filter away.
LABEL = "pr-lens.sandbox=1"

STATUS_MARKER = "pr-lens-sandbox-status"

# SIGKILL is not negotiable, but docker still has to notice and tear the container down.
KILL_WAIT_SECONDS = 20

# Where the Python setup step installs the project, and so where its console scripts land.
# In /tmp rather than under /work so the build's package discovery never finds it.
PROJECT_SITE = "/tmp/pr-lens-site"

# 137 is 128 + SIGKILL, what docker run reports for a container killed from outside.
_SIGKILL_EXIT = 137

# Bind mounts are passed through --mount, which docker parses as CSV. A path it could
# split is refused rather than quoted.
_SAFE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")

# The leading newline matters: pip ends its coloured output without one, and a status line
# glued to the end of someone else's line is a status line nobody finds. Seen 14 Sep 2026.
_REPORT = rf"""
report() {{
  local oom peak
  oom=$(sed -n 's/^oom_kill //p' /sys/fs/cgroup/memory.events 2>/dev/null)
  peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)
  printf '\n%s exit=%s setup=%s oom_kill=%s peak=%s\n' \
    {STATUS_MARKER} "$1" "$2" "${{oom:-}}" "${{peak:-}}" >&2
}}
"""

# Arguments: the number of setup words, the setup words, then the command. A failed copy
# exits without a status line, which reads as a sandbox error, and cp says why on stderr.
_RUN_SCRIPT = (
    "set -u\n"
    + _REPORT
    + rf"""
cp -R {SOURCE_MOUNT}/. {WORK_DIR}/ || exit 70
if [ -d {DEPS_MOUNT}/work ]; then
  while IFS= read -r -d '' dir; do
    rel=${{dir#{DEPS_MOUNT}/work/}}
    rm -rf "{WORK_DIR}/$rel"
    ln -s "$dir" "{WORK_DIR}/$rel"
  done < <(find {DEPS_MOUNT}/work -name node_modules -type d -prune -print0)
fi
if [ -d {DEPS_MOUNT}/venv ]; then
  export VIRTUAL_ENV={DEPS_MOUNT}/venv
fi
export PATH="{PROJECT_SITE}/bin:{DEPS_MOUNT}/venv/bin:$PATH"
cd {WORK_DIR}
setup_words=$1
shift
setup=-
if [ "$setup_words" -gt 0 ]; then
  "${{@:1:$setup_words}}" >&2
  setup=$?
  shift "$setup_words"
fi
"$@"
status=$?
report "$status" "$setup"
exit "$status"
"""
)

# Arguments: the fetcher, then pip requirements. Project config files are deleted from the
# copy before a JS install because they can point the registry elsewhere or, for yarn and
# pnpm, name repository code to run as part of the install itself.
_FETCH_SCRIPT = (
    "set -u\n"
    + _REPORT
    + rf"""
fetch() {{
  set -e
  fetcher=$1
  shift
  case "$fetcher" in
    pip)
      python -m venv --system-site-packages {DEPS_MOUNT}/venv
      if [ "$#" -gt 0 ]; then
        {DEPS_MOUNT}/venv/bin/python -m pip install --disable-pip-version-check \
          --no-input --only-binary=:all: -- "$@"
      fi
      ;;
    npm-ci|npm-install|pnpm|yarn)
      mkdir -p {DEPS_MOUNT}/work
      cp -R {SOURCE_MOUNT}/. {DEPS_MOUNT}/work/
      cd {DEPS_MOUNT}/work
      rm -f .npmrc .yarnrc .yarnrc.yml .pnpmfile.cjs
      case "$fetcher" in
        npm-ci) npm ci --ignore-scripts --no-audit --no-fund ;;
        npm-install) npm install --ignore-scripts --no-audit --no-fund --no-package-lock ;;
        pnpm) pnpm install --frozen-lockfile --ignore-scripts --ignore-pnpmfile \
          --store-dir {DEPS_MOUNT}/.pnpm-store ;;
        yarn) yarn install --frozen-lockfile --ignore-scripts --non-interactive \
          --cache-folder {DEPS_MOUNT}/.yarn-cache ;;
      esac
      ;;
    *)
      echo "unknown fetcher $fetcher" >&2
      return 64
      ;;
  esac
}}
( fetch "$@" )
status=$?
report "$status" -
exit "$status"
"""
)

_STATUS_LINE = re.compile(
    rf"^{STATUS_MARKER} exit=(?P<exit>\d+) setup=(?P<setup>-|\d+) "
    r"oom_kill=(?P<oom>\d*) peak=(?P<peak>\d*)$",
    re.MULTILINE,
)

# Set for every step. A test run that reads HOME or writes a cache gets /tmp, and CI=true
# is what turns jest and vitest from watch mode into a single run.
_BASE_ENV: tuple[tuple[str, str], ...] = (
    ("HOME", "/tmp"),
    ("XDG_CACHE_HOME", "/tmp/.cache"),
    ("CI", "true"),
    ("NO_COLOR", "1"),
    ("FORCE_COLOR", "0"),
)

# The fetch step's environment is fixed here, not by the caller. Package manager caches go
# on the volume, which is disk, not on /tmp, which is memory.
_FETCH_ENV: tuple[tuple[str, str], ...] = (
    *_BASE_ENV,
    ("npm_config_cache", f"{DEPS_MOUNT}/.npm-cache"),
    ("npm_config_update_notifier", "false"),
    ("npm_config_manage_package_manager_versions", "false"),
    ("COREPACK_ENABLE_STRICT", "0"),
    ("PIP_NO_INPUT", "1"),
    ("PIP_NO_COLOR", "1"),
    # On the volume, so a fetch retried after dropping a requirement does not download
    # everything again.
    ("PIP_CACHE_DIR", f"{DEPS_MOUNT}/.pip-cache"),
)

# How many requirements a fetch may drop before it gives up and reports the failure.
MAX_DROPPED_REQUIREMENTS = 10

_NO_DISTRIBUTION = re.compile(r"No matching distribution found for (\S+)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class SandboxError(RuntimeError):
    """Docker itself failed: a pull, a volume. Not how a repository's tests went."""


def container_name() -> str:
    """Unique per run, so two sandboxes on one pull request never address each other."""
    return f"pr-lens-sandbox-{uuid.uuid4().hex[:16]}"


def _source_mount(source_dir: Path) -> str:
    path = source_dir.resolve()
    if not path.is_dir():
        raise SpecError(f"source_dir {source_dir} is not a directory")
    if not _SAFE_PATH.match(str(path)):
        raise SpecError(f"source_dir {path} has characters --mount would misread")
    return f"type=bind,source={path},target={SOURCE_MOUNT},readonly"


def _env_args(env: tuple[tuple[str, str], ...]) -> list[str]:
    # Always NAME=value. A bare NAME would copy the variable from the job's environment,
    # which is where the Actions secrets are.
    merged = dict(_BASE_ENV)
    merged.update(env)
    args: list[str] = []
    for name, value in merged.items():
        args += ["--env", f"{name}={value}"]
    return args


def _hardening(name: str) -> list[str]:
    """Everything the two steps share. Network is the one thing decided per step."""
    return [
        "--rm",
        "--init",
        "--name",
        name,
        "--label",
        LABEL,
        "--user",
        RUN_AS_USER,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        CPU_LIMIT,
        "--memory",
        f"{MEMORY_LIMIT_MB}m",
        "--memory-swap",
        f"{MEMORY_LIMIT_MB}m",
        "--pids-limit",
        str(PIDS_LIMIT),
        "--tmpfs",
        f"{WORK_DIR}:rw,exec,nosuid,nodev,size={WORK_TMPFS_MB}m,mode=1777",
        "--tmpfs",
        f"/tmp:rw,exec,nosuid,nodev,size={TMP_TMPFS_MB}m,mode=1777",
        # The default json-file driver writes everything the container prints to the
        # runner's disk, uncapped. The attached streams are all this module reads.
        "--log-driver",
        "none",
    ]


def run_argv(spec: SandboxSpec, name: str) -> list[str]:
    """The docker command for a step that runs repository code. It has no network."""
    argv = [DOCKER, "run", *_hardening(name), "--network", "none"]
    argv += ["--mount", _source_mount(spec.source_dir)]
    if spec.deps_volume is not None:
        argv += ["--mount", f"type=volume,source={spec.deps_volume},target={DEPS_MOUNT},readonly"]
    argv += ["--workdir", WORK_DIR, *_env_args(spec.env)]
    argv += [spec.image, "bash", "-c", _RUN_SCRIPT, "pr-lens-sandbox"]
    argv += [str(len(spec.setup)), *spec.setup, *spec.command]
    return argv


def fetch_argv(spec: FetchSpec, name: str) -> list[str]:
    """The docker command for the one networked step. Its command is fixed above."""
    argv = [DOCKER, "run", *_hardening(name), "--network", "bridge"]
    argv += ["--mount", _source_mount(spec.source_dir)]
    argv += ["--mount", f"type=volume,source={spec.deps_volume},target={DEPS_MOUNT}"]
    argv += ["--workdir", WORK_DIR, *_env_args(_FETCH_ENV)]
    argv += [spec.image, "bash", "-c", _FETCH_SCRIPT, "pr-lens-fetch"]
    argv += [spec.fetcher.value, *spec.requirements]
    return argv


class _Capture:
    """A capped stream. Keeps the head and the tail, because the head says what ran and
    the tail carries the summary and the status line."""

    def __init__(self, cap: int) -> None:
        self._head_cap = cap // 4
        self._tail_cap = cap - self._head_cap
        self._head = bytearray()
        self._tail = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> None:
        room = self._head_cap - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            excess = len(self._tail) - self._tail_cap
            if excess > 0:
                del self._tail[:excess]
                self.dropped += excess

    async def drain(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while chunk := await stream.read(65536):
            self.feed(chunk)

    def text(self) -> str:
        head = self._head.decode("utf-8", errors="replace")
        tail = self._tail.decode("utf-8", errors="replace")
        if self.dropped:
            return f"{head}\n[pr-lens: {self.dropped} bytes dropped]\n{tail}"
        return head + tail


@dataclass(frozen=True, slots=True)
class _Status:
    exit_code: int
    setup_exit_code: int | None
    oom_kills: int
    memory_peak_bytes: int | None


def _read_status(stderr: str) -> tuple[_Status | None, str]:
    """The last status line, and stderr without it. Last, because the suite under test
    can print anything it likes, including a line that looks like this one."""
    matches = list(_STATUS_LINE.finditer(stderr))
    if not matches:
        return None, stderr
    match = matches[-1]
    status = _Status(
        exit_code=int(match["exit"]),
        setup_exit_code=None if match["setup"] == "-" else int(match["setup"]),
        oom_kills=int(match["oom"] or 0),
        memory_peak_bytes=int(match["peak"]) if match["peak"] else None,
    )
    before = stderr[: match.start()].removesuffix("\n")
    return status, before + stderr[match.end() + 1 :]


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        DOCKER, *args, stdin=DEVNULL, stdout=PIPE, stderr=PIPE
    )
    stdout, stderr = await proc.communicate()
    code = proc.returncode if proc.returncode is not None else -1
    return code, (stdout if code == 0 else stderr).decode("utf-8", errors="replace").strip()


async def _remove(name: str) -> None:
    code, message = await _docker("rm", "--force", name)
    if code != 0 and "No such container" not in message:
        logger.warning("could not remove container %s: %s", name, message)


async def execute(argv: list[str], name: str, timeout_seconds: int) -> SandboxResult:
    """Run one docker command to completion or to the wall clock, and leave nothing behind.

    Public because the escape test's positive control has to run through exactly this
    path with exactly one flag removed, or it proves nothing about the path that ships.
    """
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(*argv, stdin=DEVNULL, stdout=PIPE, stderr=PIPE)
    stdout, stderr = _Capture(OUTPUT_CAP_BYTES), _Capture(OUTPUT_CAP_BYTES)
    readers = asyncio.gather(stdout.drain(proc.stdout), stderr.drain(proc.stderr))
    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout_seconds)
    except TimeoutError:
        timed_out = True
    finally:
        # Also reached on cancellation, which is how a cancelled Actions job would
        # otherwise leave a container running for the rest of the runner's life.
        if proc.returncode is None:
            await _docker("kill", "--signal", "KILL", name)
            try:
                await asyncio.wait_for(proc.wait(), KILL_WAIT_SECONDS)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        await _remove(name)
    await readers
    duration = time.monotonic() - started

    status, stderr_text = _read_status(stderr.text())
    docker_exit = proc.returncode
    oom_killed = False
    if status is not None:
        oom_killed = status.oom_kills > 0
    elif not timed_out and docker_exit == _SIGKILL_EXIT:
        oom_killed = True

    return SandboxResult(
        exit_code=None if status is None else status.exit_code,
        stdout=stdout.text(),
        stderr=stderr_text,
        duration_seconds=duration,
        timed_out=timed_out,
        oom_killed=oom_killed,
        setup_exit_code=None if status is None else status.setup_exit_code,
        memory_peak_bytes=None if status is None else status.memory_peak_bytes,
        docker_exit_code=docker_exit,
        stdout_truncated=stdout.dropped > 0,
        stderr_truncated=stderr.dropped > 0,
    )


async def run(spec: SandboxSpec) -> SandboxResult:
    name = container_name()
    return await execute(run_argv(spec, name), name, spec.timeout_seconds)


async def fetch(spec: FetchSpec) -> SandboxResult:
    name = container_name()
    return await execute(fetch_argv(spec, name), name, spec.timeout_seconds)


def unavailable_requirement(output: str, requirements: tuple[str, ...]) -> str | None:
    """The requirement pip said has no distribution it may use, if it is one we passed.

    Only a top-level requirement is dropped. When the missing package came in as someone
    else's dependency this returns None and the fetch is reported as failed, because
    guessing which requirement pulled it in would be guessing.
    """
    match = _NO_DISTRIBUTION.search(_ANSI.sub("", output))
    if match is None:
        return None
    try:
        missing = canonicalize_name(Requirement(match[1]).name)
    except InvalidRequirement:
        return None
    for line in requirements:
        if canonicalize_name(Requirement(line).name) == missing:
            return line
    return None


async def fetch_dropping_unavailable(spec: FetchSpec) -> tuple[SandboxResult, tuple[str, ...]]:
    """fetch, and when pip fails on one requirement with no wheel, drop it and resolve again.

    Binary-only means a single sdist-only package fails the whole resolve. Measured on the
    mined repositories 14 Sep 2026, that was four of the first sixteen: a docs plugin in
    httpx's requirements, pyspark in pandera's extras. Dropping it and saying so runs every
    test that does not need it. Returns what was dropped, for the report.
    """
    dropped: list[str] = []
    current = spec
    while True:
        result = await fetch(current)
        if result.outcome is Outcome.PASSED or len(dropped) >= MAX_DROPPED_REQUIREMENTS:
            return result, tuple(dropped)
        culprit = unavailable_requirement(result.stdout + result.stderr, current.requirements)
        if culprit is None:
            return result, tuple(dropped)
        dropped.append(culprit)
        remaining = tuple(line for line in current.requirements if line != culprit)
        current = replace(current, requirements=remaining)


async def pull(image: str) -> float:
    """Seconds to pull the image. On a cold runner this is latency nobody budgets for."""
    started = time.monotonic()
    code, message = await _docker("pull", "--quiet", image)
    if code != 0:
        raise SandboxError(f"docker pull {image} failed: {message}")
    return time.monotonic() - started


@asynccontextmanager
async def deps_volume() -> AsyncIterator[str]:
    """A fresh volume for one run's dependencies, removed however the run ends."""
    name = f"pr-lens-{uuid.uuid4().hex[:16]}"
    code, message = await _docker("volume", "create", "--label", LABEL, name)
    if code != 0:
        raise SandboxError(f"docker volume create failed: {message}")
    try:
        yield name
    finally:
        code, message = await _docker("volume", "rm", "--force", name)
        if code != 0:
            logger.warning("could not remove volume %s: %s", name, message)
