"""Every workflow runs in a concurrency group, and review runs one per pull request.

Read as text rather than parsed as YAML, which keeps a YAML library out of the test
dependencies for a check this small. A top-level key starts at column zero, so the
patterns below cannot be satisfied by a nested key or a comment.
"""

import re
from pathlib import Path

import pytest

from pr_lens.jobs.plan import MATRICES

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_every_workflow_has_a_concurrency_group(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert re.search(r"^concurrency:\s*$", text, re.MULTILINE), f"{path.name} has no group"
    assert re.search(r"^  group: \S", text, re.MULTILINE), f"{path.name} names no group"


def test_review_runs_once_per_pull_request_and_the_newest_wins() -> None:
    """The live bug from the 12 Sep audit: two pushes to one PR reviewed it twice."""
    text = workflow("review.yml")
    group = re.search(r"^  group: (?P<key>.+)$", text, re.MULTILINE)
    assert group is not None
    assert "client_payload.repo_full_name" in group["key"]
    assert "client_payload.pr_number" in group["key"]
    # Not the delivery id, which differs per push and would put every run in its own group.
    assert "delivery_id" not in group["key"]
    assert re.search(r"^  cancel-in-progress: true\s*$", text, re.MULTILINE)


def test_every_phase_2_matrix_fits_in_one_workflow_run() -> None:
    """GitHub refuses a matrix of more than 256 jobs. Better to fail here than at dispatch."""
    for build in MATRICES.values():
        assert 0 < len(build()) <= 256


def test_no_workflow_caches_the_private_corpus() -> None:
    """The hub cache holds whatever the job downloaded, private corpus shards included, and
    caches on a public repo can be restored by pull request workflows. Cache models by name."""
    for path in WORKFLOWS.glob("*.yml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not re.search(r"\.cache/huggingface/hub/?\s*$", line), f"{path.name}: {line}"
