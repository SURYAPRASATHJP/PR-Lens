"""The Phase 3 gate: a passing test here proves network is unreachable inside the sandbox.

Run in sandbox.yml on a GitHub-hosted runner against the digests in
pr_lens.sandbox.images, which are the images every later run pulls. Marked docker and
skipped where there is no daemon, except that sandbox.yml sets PR_LENS_REQUIRE_DOCKER, so
on the runner a missing daemon fails the gate rather than skipping it green.

Three ways an escape test passes for the wrong reason, and what stops each here:

- The probe tool is missing. curl in an image without curl fails with or without a
  network. Each image is probed with its own runtime, python in one and node in the
  other, which cannot be absent from an image built to run them.
- The failure is the wrong failure. A missing resolver is not isolation. DNS, a TCP
  connect to a literal IP and an HTTP request to that IP are each asserted to fail with
  the specific error that means "no route": EAI_AGAIN and ENETUNREACH, measured on
  14 Sep 2026. The interface list is asserted to be loopback alone.
- The probe is broken. The positive control runs the identical argv, through the
  identical code path, with only the --network none pair removed, and asserts all three
  probes succeed. Without it this suite could not tell isolation from a probe that never
  worked, and it would keep passing after someone deleted the flag.
"""

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pr_lens.sandbox import images, runner
from pr_lens.sandbox.detect import detect
from pr_lens.sandbox.spec import (
    PIDS_LIMIT,
    Fetcher,
    FetchSpec,
    Outcome,
    SandboxResult,
    SandboxSpec,
)

pytestmark = pytest.mark.docker

REQUIRE_DOCKER = os.environ.get("PR_LENS_REQUIRE_DOCKER") == "1"

# For a laptop whose daemon cannot run the published linux/amd64 images. sandbox.yml never
# sets these, tests/test_workflows.py holds that, and with PR_LENS_REQUIRE_DOCKER set an
# override fails the suite outright.
PYTHON = os.environ.get("PR_LENS_TEST_PYTHON_IMAGE", images.PYTHON_IMAGE)
NODE = os.environ.get("PR_LENS_TEST_NODE_IMAGE", images.NODE_IMAGE)

PYTHON_PROBE = r"""
import errno, http.client, json, os, socket

def name(error):
    if isinstance(error, socket.gaierror):
        codes = {getattr(socket, n): n for n in dir(socket) if n.startswith("EAI_")}
        return codes.get(error.errno, str(error.errno))
    return errno.errorcode.get(error.errno, str(error.errno))

out = {"interfaces": sorted(os.listdir("/sys/class/net"))}
try:
    socket.getaddrinfo("github.com", 443)
    out["dns"] = "ok"
except OSError as error:
    out["dns"] = ["error", name(error)]
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5).close()
    out["tcp"] = "ok"
except OSError as error:
    out["tcp"] = ["error", name(error)]
try:
    connection = http.client.HTTPConnection("1.1.1.1", 80, timeout=5)
    connection.request("HEAD", "/")
    out["http"] = connection.getresponse().status
except OSError as error:
    out["http"] = ["error", name(error)]
print(json.dumps(out))
"""

NODE_PROBE = r"""
const dns = require('node:dns').promises;
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const tcp = () => new Promise((done) => {
  const socket = net.connect({ host: '1.1.1.1', port: 443 });
  socket.setTimeout(5000, () => { socket.destroy(); done(['error', 'TIMEOUT']); });
  socket.on('connect', () => { socket.destroy(); done('ok'); });
  socket.on('error', (error) => done(['error', error.code]));
});
const web = () => new Promise((done) => {
  const request = http.request({ host: '1.1.1.1', port: 80, method: 'HEAD', path: '/' });
  request.setTimeout(5000, () => { request.destroy(); done(['error', 'TIMEOUT']); });
  request.on('response', (response) => { response.resume(); done(response.statusCode); });
  request.on('error', (error) => done(['error', error.code]));
  request.end();
});
(async () => {
  const out = { interfaces: fs.readdirSync('/sys/class/net').sort() };
  try { await dns.lookup('github.com'); out.dns = 'ok'; }
  catch (error) { out.dns = ['error', error.code]; }
  out.tcp = await tcp();
  out.http = await web();
  console.log(JSON.stringify(out));
})();
"""

PROBES = {
    "python": (PYTHON, ("python", "-c", PYTHON_PROBE)),
    "node": (NODE, ("node", "-e", NODE_PROBE)),
}

ISOLATED = {
    "interfaces": ["lo"],
    "dns": ["error", "EAI_AGAIN"],
    "tcp": ["error", "ENETUNREACH"],
    "http": ["error", "ENETUNREACH"],
}


def _docker_answers() -> bool:
    if shutil.which("docker") is None:
        return False
    info = subprocess.run(["docker", "info"], capture_output=True, check=False)  # noqa: S607
    return info.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def docker() -> Iterator[None]:
    if not _docker_answers():
        if REQUIRE_DOCKER:
            pytest.fail("PR_LENS_REQUIRE_DOCKER is set and there is no Docker daemon")
        pytest.skip("no Docker daemon; the escape suite runs in sandbox.yml")
    if REQUIRE_DOCKER and (PYTHON, NODE) != (images.PYTHON_IMAGE, images.NODE_IMAGE):
        pytest.fail("the gate must run against the pinned images, not an override")
    for image in (PYTHON, NODE):
        asyncio.run(runner.pull(image))
    yield


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A checkout anyone may write to, so a refused write is the mount's doing alone."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "README.md").write_text("original\n")
    checkout.chmod(0o777)
    (checkout / "README.md").chmod(0o666)
    return checkout


def probe_output(result: SandboxResult) -> Any:
    """The last line a probe printed, as JSON. Every probe here prints exactly one."""
    assert result.outcome is Outcome.PASSED, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def without_network(argv: list[str]) -> list[str]:
    """The positive control's argv. Exactly one change: the --network none pair is gone."""
    index = argv.index("--network")
    assert argv[index + 1] == "none"
    return argv[:index] + argv[index + 2 :]


# The gate


@pytest.mark.parametrize("language", sorted(PROBES))
async def test_network_is_unreachable_inside_the_container(language: str, source: Path) -> None:
    image, command = PROBES[language]
    spec = SandboxSpec(image=image, command=command, source_dir=source, timeout_seconds=60)

    isolated = probe_output(await runner.run(spec))
    assert isolated == ISOLATED

    name = runner.container_name()
    argv = runner.run_argv(spec, name)
    control_argv = without_network(argv)
    index = argv.index("--network")
    assert control_argv == argv[:index] + argv[index + 2 :]
    assert "--network" not in control_argv

    control = probe_output(await runner.execute(control_argv, name, 60))
    assert "eth0" in control["interfaces"]
    assert control["dns"] == "ok"
    assert control["tcp"] == "ok"
    assert isinstance(control["http"], int)
    # The control is what run_argv would produce if --network none were deleted from it.
    # Every probe differs from ISOLATED, so that deletion fails the first assertion above.
    assert all(control[key] != ISOLATED[key] for key in ISOLATED)


# Beside the gate


async def test_the_checkout_the_root_and_the_dependencies_are_read_only(source: Path) -> None:
    script = r"""
import errno, json
out = {}
paths = ("/src/README.md", "/src/new", "/new", "/usr/new", "/deps/new", "/work/new", "/tmp/new")
for path in paths:
    try:
        with open(path, "w") as handle:
            handle.write("changed")
        out[path] = "written"
    except OSError as error:
        out[path] = errno.errorcode[error.errno]
print(json.dumps(out))
"""
    async with runner.deps_volume() as volume:
        result = await runner.run(
            SandboxSpec(
                image=PYTHON,
                command=("python", "-c", script),
                source_dir=source,
                deps_volume=volume,
                timeout_seconds=60,
            )
        )
    assert probe_output(result) == {
        "/src/README.md": "EROFS",
        "/src/new": "EROFS",
        "/new": "EROFS",
        "/usr/new": "EROFS",
        "/deps/new": "EROFS",
        "/work/new": "written",
        "/tmp/new": "written",
    }
    assert sorted(path.name for path in source.iterdir()) == ["README.md"]
    assert (source / "README.md").read_text() == "original\n"


async def test_the_run_works_from_a_writable_copy_of_the_checkout(source: Path) -> None:
    """The read-only decision's other half: suites that write beside their source still run."""
    (source / "test_writes.py").write_text(
        "from pathlib import Path\n\n"
        "def test_writes_beside_itself():\n"
        "    Path(__file__).with_name('scratch.txt').write_text('ok')\n"
    )
    result = await runner.run(
        SandboxSpec(
            image=PYTHON,
            command=("python", "-m", "pytest", "-q"),
            source_dir=source,
            timeout_seconds=120,
        )
    )
    assert result.outcome is Outcome.PASSED, result.stdout + result.stderr
    assert not (source / "scratch.txt").exists()
    assert not (source / ".pytest_cache").exists()


async def test_the_process_inside_is_not_root_and_holds_no_capabilities(source: Path) -> None:
    script = (
        "import json\n"
        "status = dict(line.split(':', 1) for line in open('/proc/self/status'))\n"
        "print(json.dumps({k: status[k].split() for k in "
        "('Uid', 'Gid', 'CapEff', 'CapPrm', 'CapBnd', 'NoNewPrivs')}))\n"
    )
    result = await runner.run(
        SandboxSpec(image=PYTHON, command=("python", "-c", script), source_dir=source)
    )
    status = probe_output(result)
    assert status["Uid"] == ["65534"] * 4
    assert status["Gid"] == ["65534"] * 4
    assert status["CapEff"] == status["CapPrm"] == ["0000000000000000"]
    assert status["CapBnd"] == ["0000000000000000"]
    assert status["NoNewPrivs"] == ["1"]


async def test_a_fork_bomb_hits_the_pid_cap(source: Path) -> None:
    script = r"""
import errno, json, os, signal, time
children = []
failure = None
while len(children) < 4 * LIMIT:
    try:
        pid = os.fork()
    except OSError as error:
        failure = errno.errorcode[error.errno]
        break
    if pid == 0:
        time.sleep(60)
        os._exit(0)
    children.append(pid)
for pid in children:
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
print(json.dumps({"forked": len(children), "failure": failure}))
""".replace("LIMIT", str(PIDS_LIMIT))
    result = await runner.run(
        SandboxSpec(image=PYTHON, command=("python", "-c", script), source_dir=source)
    )
    report = probe_output(result)
    assert report["failure"] == "EAGAIN"
    assert 0 < report["forked"] < PIDS_LIMIT


async def test_an_oom_killed_test_is_reported_as_oom_not_as_a_failure(source: Path) -> None:
    """The runner process survives and exits 1, as pytest does when a worker dies. Only the
    cgroup knows the worker was killed for memory, and the result says so."""
    allocate = "block = b'x' * (3 * 1024 ** 3)"
    script = (
        "import subprocess, sys\n"
        f"subprocess.run([sys.executable, '-c', {allocate!r}])\n"
        "sys.exit(1)\n"
    )
    result = await runner.run(
        SandboxSpec(image=PYTHON, command=("python", "-c", script), source_dir=source)
    )
    assert result.exit_code == 1
    assert result.oom_killed
    assert result.outcome is Outcome.OOM_KILLED
    assert result.memory_peak_bytes is not None and result.memory_peak_bytes > 1024**3


async def test_filling_the_work_tree_is_enospc_long_before_the_memory_cap(source: Path) -> None:
    script = (
        "import errno, json\n"
        "try:\n"
        "    with open('/work/big', 'wb') as handle:\n"
        "        for _ in range(2048):\n"
        "            handle.write(b'x' * 1024 ** 2)\n"
        "    print(json.dumps('written'))\n"
        "except OSError as error:\n"
        "    print(json.dumps(errno.errorcode[error.errno]))\n"
    )
    result = await runner.run(
        SandboxSpec(image=PYTHON, command=("python", "-c", script), source_dir=source)
    )
    assert probe_output(result) == "ENOSPC"
    assert not result.oom_killed


async def test_the_wall_clock_kills_the_container_and_leaves_nothing(source: Path) -> None:
    spec = SandboxSpec(image=PYTHON, command=("sleep", "600"), source_dir=source, timeout_seconds=3)
    name = runner.container_name()
    result = await runner.execute(runner.run_argv(spec, name), name, spec.timeout_seconds)
    assert result.outcome is Outcome.TIMED_OUT
    assert result.duration_seconds < 3 + runner.KILL_WAIT_SECONDS
    process = await asyncio.create_subprocess_exec(
        "docker", "ps", "--all", "--quiet", "--filter", f"name=^{name}$",
        stdout=asyncio.subprocess.PIPE,
    )  # fmt: skip
    left, _ = await process.communicate()
    assert process.returncode == 0
    assert left.decode().strip() == ""


async def test_two_runs_at_once_share_no_writable_path(source: Path) -> None:
    script = (
        "import json, os, sys, time\n"
        "me = sys.argv[1]\n"
        "for directory in ('/work', '/tmp'):\n"
        "    open(os.path.join(directory, 'marker-' + me), 'w').close()\n"
        "time.sleep(3)\n"
        "found = [n for d in ('/work', '/tmp') for n in os.listdir(d) if n.startswith('marker-')]\n"
        "print(json.dumps(sorted(found)))\n"
    )

    async def one(label: str) -> SandboxResult:
        return await runner.run(
            SandboxSpec(image=PYTHON, command=("python", "-c", script, label), source_dir=source)
        )

    first, second = await asyncio.gather(one("a"), one("b"))
    assert probe_output(first) == ["marker-a", "marker-a"]
    assert probe_output(second) == ["marker-b", "marker-b"]
    assert sorted(path.name for path in source.iterdir()) == ["README.md"]


async def test_the_fetch_step_never_runs_a_lifecycle_script(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    marker = "require('fs').writeFileSync('/deps/ran-' + process.env.npm_lifecycle_event, '')"
    scripts = dict.fromkeys(
        ("preinstall", "install", "postinstall", "prepare"), f'node -e "{marker}"'
    )
    (project / "package.json").write_text(
        json.dumps({"name": "demo", "version": "1.0.0", "scripts": scripts})
    )
    (project / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.0.0",
                "lockfileVersion": 3,
                "requires": True,
                "packages": {"": {"name": "demo", "version": "1.0.0", "hasInstallScript": True}},
            }
        )
    )
    async with runner.deps_volume() as volume:
        fetched = await runner.fetch(
            FetchSpec(image=NODE, fetcher=Fetcher.NPM_CI, source_dir=project, deps_volume=volume)
        )
        assert fetched.outcome is Outcome.PASSED, fetched.stdout + fetched.stderr
        listing = await runner.run(
            SandboxSpec(
                image=NODE,
                command=(
                    "node",
                    "-e",
                    "console.log(JSON.stringify(require('fs').readdirSync('/deps')))",
                ),
                source_dir=project,
                deps_volume=volume,
            )
        )
    assert not [name for name in probe_output(listing) if name.startswith("ran-")]


async def test_a_python_project_builds_and_tests_with_no_network(tmp_path: Path) -> None:
    """The two steps together: pip fetches wheels with network, then the project's own
    build backend runs with none, and importlib.metadata can see the result."""
    project = tmp_path / "project"
    (project / "src" / "demo").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
    )
    (project / "src" / "demo" / "__init__.py").write_text(
        "import importlib.metadata\nVERSION = importlib.metadata.version('demo')\n"
    )
    (project / "tests" / "test_demo.py").write_text(
        "import demo\n\ndef test_version():\n    assert demo.VERSION == '0.1.0'\n"
    )
    detection = detect(project)
    assert detection.fetcher is Fetcher.PIP
    async with runner.deps_volume() as volume:
        fetched = await runner.fetch(
            FetchSpec(PYTHON, Fetcher.PIP, project, volume, detection.requirements)
        )
        assert fetched.outcome is Outcome.PASSED, fetched.stderr
        result = await runner.run(
            SandboxSpec(
                image=PYTHON,
                command=detection.command,
                source_dir=project,
                setup=detection.setup,
                deps_volume=volume,
                env=detection.env,
            )
        )
    assert result.setup_exit_code == 0, result.stderr
    assert result.outcome is Outcome.PASSED, result.stdout


async def _labelled(kind: str) -> list[str]:
    process = await asyncio.create_subprocess_exec(
        "docker",
        kind,
        "ls",
        "--quiet",
        "--filter",
        f"label={runner.LABEL}",
        *(["--all"] if kind == "container" else []),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    return stdout.decode().split()


async def test_the_suite_left_no_container_and_no_volume_behind() -> None:
    """Last in the file on purpose: every test above created containers, and some created
    volumes, including the one that was killed by the wall clock."""
    assert await _labelled("container") == []
    assert await _labelled("volume") == []
