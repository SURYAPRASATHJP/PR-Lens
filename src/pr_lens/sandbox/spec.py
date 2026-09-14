"""What a sandbox run is allowed to be.

The isolation rules live here as properties of the types, not as habits of the caller.

There is no network field on SandboxSpec. Not a field defaulting to off, not an enum with
one safe member. A later caller cannot ask for a networked sandbox because there is no way
to express the request, and a refactor cannot quietly weaken isolation because there is
nothing to set. Every container that executes code from the repository under review, its
tests, its build backend, its conftest.py, is built from a SandboxSpec. The escape test in
tests/test_sandbox_isolation.py is what proves the resulting container has no network.

Network exists in exactly one other place, FetchSpec, and only because a container with
no network cannot install a dependency. What makes that safe is that FetchSpec has no
command field either. It names a package manager from a closed set, and runner.py turns
that into a fixed invocation which executes nothing from the repository or from what it
downloads: pip with --only-binary=:all:, so no sdist's setup.py ever runs, and npm, pnpm
and yarn with --ignore-scripts, so no lifecycle script does. The fetch step downloads
files. Everything that runs them runs later, with no network.

Images are pinned by digest. A tag is mutable, so a sandbox pinned to a tag runs
something other than what the escape test proved, which would make the gate a statement
about the past rather than about production.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

# The public-repo runner is 4 vCPU and 16 GB. Half the cores leaves room for the job itself
# and for a second sandbox beside this one, which is the concurrent case, not the edge case.
CPU_LIMIT = "2.0"

# Suites that need more than this are not suites we can serve on free infrastructure, and
# an honest OOM is a better answer than a runner that thrashes for five minutes. Swap is
# set to the same figure, which means none: a container that swaps is slow, not bounded.
MEMORY_LIMIT_MB = 2048

# Both writable trees are tmpfs, and tmpfs pages are charged to the container's memory
# cgroup. Measured 14 Sep 2026: a tmpfs larger than the memory cap does not fill up, it
# gets every process in the container OOM-killed, the reporting wrapper included. Sized
# together well under the cap, a suite that writes too much gets ENOSPC, which it can
# report, and still has 1 GB of memory to run in. tests/test_sandbox_spec.py holds the sum.
WORK_TMPFS_MB = 768
TMP_TMPFS_MB = 256

# A fork bomb is the cheapest escape attempt there is, and without this it takes the runner
# down rather than the container. 256 is comfortably above what a test suite spawns,
# pytest-xdist and jest workers included.
PIDS_LIMIT = 256

# Five minutes for the tests. Long enough for a real suite on a small repository, short
# enough that a hung test does not spend a review's whole latency budget.
WALL_TIMEOUT_SECONDS = 300

# Ten for the fetch, which is downloads rather than work and is dominated by a cold
# resolver. A dependency set that takes longer is itself worth reporting.
FETCH_TIMEOUT_SECONDS = 600

# Captured output per stream. Test output is occasionally enormous, a failing snapshot
# suite or a stack trace per assertion, and an unbounded read is how a job runs the runner
# out of memory while trying to protect it from the container.
OUTPUT_CAP_BYTES = 256_000

# A requirement list longer than this is not a test environment, it is a distribution.
MAX_REQUIREMENTS = 300

# The container's view of the world. The checkout is mounted read-only where nothing ever
# writes, and the run copies it into WORK_DIR, a tmpfs. See runner.py for why a literally
# read-only working tree does not survive contact with real test suites.
SOURCE_MOUNT = "/src"
WORK_DIR = "/work"

# The fetch step's output: a Python venv or a tree carrying node_modules. A docker volume
# on the runner's disk rather than tmpfs, because dependencies are routinely larger than
# anything the memory cap could hold. Written by the fetch step, read-only to the test run.
DEPS_MOUNT = "/deps"

# nobody:nogroup. Root inside a container is not root on the host, but it is still the
# wrong default: it makes every later hardening decision an argument instead of a given.
RUN_AS_USER = "65534:65534"

# [registry[:port]/]name@sha256:<64 hex>. No tag beside the digest: docker would ignore
# it, and a reader would not.
_PINNED_IMAGE = re.compile(
    r"^(?:[a-z0-9.-]+(?::\d+)?/)?[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$"
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Outcome(StrEnum):
    """How a sandbox run ended, from the point of view of whoever asked for it.

    Timeouts and OOM kills are outcomes, not errors. A suite that hangs or eats the box is
    evidence about the repository under review, and Phase 4 should see it as such rather
    than as a crashed job. ERROR is reserved for the sandbox itself failing: docker could
    not start the container, or the run ended without reporting how.
    """

    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    OOM_KILLED = "oom_killed"
    NO_TESTS = "no_tests"
    INSTALL_FAILED = "install_failed"
    ERROR = "error"


class Fetcher(StrEnum):
    """The only commands the networked step can run. runner.py owns each invocation."""

    PIP = "pip"
    NPM_CI = "npm-ci"
    NPM_INSTALL = "npm-install"
    PNPM = "pnpm"
    YARN = "yarn"


class SpecError(ValueError):
    """The spec could not be built. Raised before anything is executed."""


def _check_image(image: str) -> None:
    if not _PINNED_IMAGE.match(image):
        raise SpecError(
            f"image {image!r} is not pinned by digest. A tag is mutable, so a tagged image "
            "runs something other than what the escape test proved. Use name@sha256:<hex>."
        )


def _check_volume(volume: str | None) -> None:
    if volume is not None and not re.fullmatch(r"pr-lens-[a-z0-9-]+", volume):
        raise SpecError(f"volume {volume!r} is not one this sandbox created")


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """One command, in one image, with no network, under the limits above.

    There is deliberately no way here to grant network access, add a capability, change the
    user, or make the source or the dependencies writable. Those are not configuration.

    setup runs before command in the same container, for work that has to execute project
    code but is not the tests, such as building the project against the fetched
    dependencies. A failed setup is recorded and the tests still run, because a suite that
    imports from the source tree often passes without it.
    """

    image: str
    command: tuple[str, ...]
    source_dir: Path
    setup: tuple[str, ...] = ()
    deps_volume: str | None = None
    # Passed as NAME=value. Nothing is inherited from the job's environment, because the
    # job's environment holds Actions secrets and the container is running someone else's
    # code.
    env: tuple[tuple[str, str], ...] = ()
    timeout_seconds: int = WALL_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        _check_image(self.image)
        _check_volume(self.deps_volume)
        if not self.command:
            raise SpecError("command is empty")
        if self.timeout_seconds <= 0:
            raise SpecError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        for name, _ in self.env:
            if not _ENV_NAME.match(name):
                raise SpecError(f"environment variable name {name!r} is not valid")


@dataclass(frozen=True, slots=True)
class FetchSpec:
    """Download dependencies into a volume. The one step with network, and no command.

    requirements are for pip only, and each one is a PEP 508 requirement on a package index.
    They come from files in the repository under review, which is to say from whoever
    opened the pull request, so anything that pip could read as an option, a path or a URL
    is refused here rather than trusted to the argument order.
    """

    image: str
    fetcher: Fetcher
    source_dir: Path
    deps_volume: str
    requirements: tuple[str, ...] = ()
    timeout_seconds: int = FETCH_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        _check_image(self.image)
        _check_volume(self.deps_volume)
        if self.requirements and self.fetcher is not Fetcher.PIP:
            raise SpecError(f"{self.fetcher} reads its lockfile; requirements are pip only")
        if len(self.requirements) > MAX_REQUIREMENTS:
            raise SpecError(f"{len(self.requirements)} requirements, over {MAX_REQUIREMENTS}")
        for line in self.requirements:
            check_requirement(line)
        if self.timeout_seconds <= 0:
            raise SpecError(f"timeout_seconds must be positive, got {self.timeout_seconds}")


def check_requirement(line: str) -> Requirement:
    """A requirement pip will fetch from an index, or SpecError saying why not."""
    if not line or line != line.strip() or line.startswith("-"):
        raise SpecError(f"requirement {line!r} could be read as a pip option")
    try:
        requirement = Requirement(line)
    except InvalidRequirement as exc:
        raise SpecError(f"requirement {line!r} is not PEP 508: {exc}") from exc
    if requirement.url is not None:
        raise SpecError(f"requirement {line!r} names a URL, not an index package")
    return requirement


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """What came back from one container. Output is already capped, see OUTPUT_CAP_BYTES.

    exit_code is the command's own exit status as the in-container wrapper reported it.
    It is None when the wrapper never reported, which happens only when the container was
    killed from outside: by the wall clock, by the kernel's OOM killer, or by docker
    failing to start it at all.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    oom_killed: bool = False
    setup_exit_code: int | None = None
    memory_peak_bytes: int | None = None
    docker_exit_code: int | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def outcome(self) -> Outcome:
        if self.timed_out:
            return Outcome.TIMED_OUT
        if self.oom_killed:
            return Outcome.OOM_KILLED
        if self.exit_code is None:
            return Outcome.ERROR
        return Outcome.PASSED if self.exit_code == 0 else Outcome.FAILED
