"""Every job imports cleanly with only the dependency groups its runner installs.

The first Phase 2 pairs run died at import in 19 seconds: it borrowed two helpers from
jobs/ingest.py, which imports the Postgres driver, and its runner rightly does not install
the db group. 220 tests missed it, because the suite installs every group, and a check
done in-process passes spuriously once any earlier test has imported the driver. So each
import here happens in a fresh interpreter.
"""

import json
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

import pr_lens.jobs

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

# The jobs that really open Postgres. Every other module in pr_lens.jobs is held to the
# rule by default, so a job added later is covered the day it is added.
DATABASE_JOBS = frozenset({"ingest", "record_delivery", "replay", "verdicts"})

# Top-level modules each optional dependency group brings, from pyproject.toml.
GROUP_MODULES = {
    "db": frozenset({"asyncpg"}),
    "ingest": frozenset({"huggingface_hub"}),
    "retrieval": frozenset(
        {"numpy", "scipy", "torch", "sentence_transformers", "transformers", "pyarrow"}
    ),
    "sandbox": frozenset({"packaging"}),
}

PROBE = (
    "import importlib, json, sys\n"
    "importlib.import_module(sys.argv[1])\n"
    "forbidden = set(sys.argv[2:])\n"
    "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in forbidden)))\n"
)


def imported(module: str, forbidden: frozenset[str]) -> list[str]:
    """Import module in a clean interpreter and return which forbidden modules came along."""
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, module, *sorted(forbidden)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    found: list[str] = json.loads(completed.stdout.strip().splitlines()[-1])
    return found


def database_free_jobs() -> list[str]:
    return sorted(
        f"pr_lens.jobs.{info.name}"
        for info in pkgutil.iter_modules(pr_lens.jobs.__path__)
        if info.name not in DATABASE_JOBS
    )


def workflow_cases() -> list[tuple[str, str, frozenset[str]]]:
    """(workflow job, module it runs, modules its installed groups do not provide).

    Only for jobs that install with --no-dev. A plain `uv sync` installs the dev group,
    which includes every other group, so nothing can be missing there.
    """
    cases: list[tuple[str, str, frozenset[str]]] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        jobs = path.read_text(encoding="utf-8").split("\njobs:\n", 1)[-1]
        for block in re.split(r"^  (?=[\w-]+:\s*$)", jobs, flags=re.MULTILINE):
            syncs = re.findall(r"uv sync[^\n]*", block)
            modules = sorted(set(re.findall(r"python -m (pr_lens\.jobs\.\w+)", block)))
            if not syncs or not modules or not all("--no-dev" in s for s in syncs):
                continue
            installed = {g for s in syncs for g in re.findall(r"--group (\w+)", s)}
            missing = frozenset().union(
                *(names for group, names in GROUP_MODULES.items() if group not in installed)
            )
            name = f"{path.stem}:{block.split(':', 1)[0].strip()}"
            cases.extend((name, module, missing) for module in modules)
    return cases


def test_the_rules_cover_every_phase_2_job() -> None:
    """If the parsing or the module listing broke, the tests below would pass on nothing."""
    covered = set(database_free_jobs())
    for job in ("pairs", "embed", "recall", "anchor", "plan"):
        assert f"pr_lens.jobs.{job}" in covered
    from_workflows = {(name, module) for name, module, _ in workflow_cases()}
    assert ("retrieval:pairs", "pr_lens.jobs.pairs") in from_workflows
    assert ("retrieval:plan", "pr_lens.jobs.plan") in from_workflows
    assert ("retrieval:rerank", "pr_lens.jobs.recall") in from_workflows


@pytest.mark.parametrize("module", database_free_jobs())
def test_a_database_free_job_never_imports_the_driver(module: str) -> None:
    assert imported(module, GROUP_MODULES["db"]) == []


@pytest.mark.parametrize(
    ("job", "module", "missing"), workflow_cases(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_each_workflow_job_imports_only_what_it_installs(
    job: str, module: str, missing: frozenset[str]
) -> None:
    assert imported(module, missing) == [], f"{job} runs {module} without installing these"
