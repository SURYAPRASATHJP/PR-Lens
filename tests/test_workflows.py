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


def test_review_mints_a_token_for_one_repository_that_cannot_push() -> None:
    """The live job's credential. Pull requests write is what posting needs; contents stays
    read, which keeps "never pushes" true at the token, and the token reaches exactly the
    repository the delivery named."""
    text = workflow("review.yml")
    assert re.search(r"^permissions:\n  contents: read\n\n", text, re.MULTILINE)
    assert "uses: actions/create-github-app-token@v3" in text
    assert "owner: ${{ steps.record.outputs.owner }}" in text
    assert "repositories: ${{ steps.record.outputs.repo_name }}" in text
    granted = re.findall(r"^\s+permission-([\w-]+): (\w+)$", text, re.MULTILINE)
    assert sorted(granted) == [("contents", "read"), ("pull-requests", "write")]
    assert "GH_INSTALLATION_TOKEN: ${{ steps.token.outputs.token }}" in text
    assert "Phase 0 stub" not in text
    # Nothing from the payload is interpolated into a shell.
    for command in re.findall(r"^\s+run: (.+)$", text, re.MULTILINE):
        assert "${{" not in command


def test_replay_holds_no_credential_that_could_post() -> None:
    """Replay never posts, and this is what makes that true rather than intended: without
    the App key it cannot mint an installation token, and without write permission its
    own token cannot comment either."""
    text = workflow("replay.yml")
    assert "GH_APP_" not in text
    assert "create-github-app-token" not in text
    assert not re.search(r"^\s+\S+:\s*write\s*$", text, re.MULTILINE)
    assert re.search(r"^permissions:\n  contents: read\n\n", text, re.MULTILINE)


def test_replay_only_starts_from_a_replay_branch_or_by_hand() -> None:
    """A push that merely touched review code must never spend the day's tokens."""
    text = workflow("replay.yml")
    on = text[text.index("\non:") : text.index("\npermissions:")]
    assert 'branches: ["replay/**"]' in on
    assert "paths:" not in on
    assert "pull_request" not in on


def test_every_phase_2_matrix_fits_in_one_workflow_run() -> None:
    """GitHub refuses a matrix of more than 256 jobs. Better to fail here than at dispatch."""
    for build in MATRICES.values():
        assert 0 < len(build()) <= 256


def test_only_the_image_build_can_write_a_package() -> None:
    """A workflow that runs repository code with packages: write could overwrite the image
    the escape test proved. Publishing is one job's business."""
    writers = sorted(
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if re.search(r"packages:\s*write", path.read_text(encoding="utf-8"))
    )
    assert writers == ["sandbox-images.yml"]


def test_the_escape_suite_runs_in_sandbox_yml_against_the_pinned_images() -> None:
    """The gate. It runs where Docker is certain, it cannot skip itself green, and it runs
    against the digests every later run pulls, never a laptop's override."""
    text = workflow("sandbox.yml")
    assert "uv run pytest -m docker" in text
    assert re.search(r'^      PR_LENS_REQUIRE_DOCKER: "1"$', text, re.MULTILINE)
    assert "PR_LENS_TEST_" not in text
    assert re.search(r"^  packages: read$", text, re.MULTILINE)
    assert "docker login ghcr.io" in text
    # The real-data matrix waits for the gate in the same run.
    assert re.search(r"^    needs: \[escape, repos\]$", text, re.MULTILINE)


def test_ci_leaves_the_docker_suite_to_sandbox_yml() -> None:
    assert 'uv run pytest -m "not docker"' in workflow("ci.yml")


def test_no_workflow_caches_the_private_corpus() -> None:
    """The hub cache holds whatever the job downloaded, private corpus shards included, and
    caches on a public repo can be restored by pull request workflows. Cache models by name."""
    for path in WORKFLOWS.glob("*.yml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not re.search(r"\.cache/huggingface/hub/?\s*$", line), f"{path.name}: {line}"
