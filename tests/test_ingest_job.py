"""The mining job's wiring: where the repo list comes from, and which token it uses."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from pr_lens.github.cache import HttpCache
from pr_lens.github.client import API_ROOT, GitHubClient, GitHubError
from pr_lens.jobs.ingest import (
    build_sink,
    main,
    mining_token,
    parse_args,
    preflight,
    resolve_repos,
)

MINING_SET = {
    "verified": "2026-09-09",
    "repos": [{"repo": "celery/celery", "stars": 1}, {"repo": "aio-libs/aiohttp", "stars": 2}],
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MINING_REPOS", "GH_MINING_TOKEN", "GH_DISPATCH_TOKEN", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_repos_come_from_the_command_line_first() -> None:
    assert resolve_repos(parse_args(["--repo", "a/b", "--repo", "c/d"])) == ["a/b", "c/d"]


def test_the_mining_set_file_is_read_in_its_own_shape(tmp_path: Path) -> None:
    # The mining set is an object with a repos key holding per-repo verified stats. It
    # is maintained outside this repository, so the job accepts that shape as given
    # rather than asking for it to be reshaped first.
    path = tmp_path / "mining-repos.json"
    path.write_text(json.dumps(MINING_SET), encoding="utf-8")

    assert resolve_repos(parse_args(["--repos-file", str(path)])) == [
        "celery/celery",
        "aio-libs/aiohttp",
    ]


def test_a_plain_list_of_names_is_also_accepted(tmp_path: Path) -> None:
    path = tmp_path / "repos.json"
    path.write_text(json.dumps(["a/b"]), encoding="utf-8")

    assert resolve_repos(parse_args(["--repos-file", str(path)])) == ["a/b"]


def test_the_environment_carries_the_list_when_nothing_else_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The workflow passes it this way, because the mining set is not in this repository.
    monkeypatch.setenv("MINING_REPOS", "a/b, c/d ,")

    assert resolve_repos(parse_args([])) == ["a/b", "c/d"]


def test_the_environment_also_takes_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINING_REPOS", json.dumps(MINING_SET))

    assert resolve_repos(parse_args([])) == ["celery/celery", "aio-libs/aiohttp"]


def test_no_repos_anywhere_is_an_empty_list_not_a_crash() -> None:
    assert resolve_repos(parse_args([])) == []


def test_a_dedicated_mining_token_wins_over_the_dispatch_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GH_DISPATCH_TOKEN", "dispatch")
    monkeypatch.setenv("GH_MINING_TOKEN", "mining")

    assert mining_token() == "mining"


def test_the_dispatch_token_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_DISPATCH_TOKEN", "dispatch")

    assert mining_token() == "dispatch"


def test_the_workflows_own_token_is_never_picked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1,000 requests/hour per repository. Reading it here would mine at a fifth of the
    # speed and look like a slow network.
    monkeypatch.setenv("GITHUB_TOKEN", "ghs-workflow")

    assert mining_token() is None


def test_the_local_sink_is_the_default(tmp_path: Path) -> None:
    sink = build_sink(parse_args(["--corpus-dir", str(tmp_path)]))

    assert sink.read_manifest("anything") == {}


def test_the_hub_sink_refuses_to_run_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        build_sink(parse_args(["--sink", "huggingface"]))


async def test_a_run_without_repos_fails_loudly() -> None:
    assert await main([]) == 1


async def test_a_run_without_a_token_fails_loudly() -> None:
    assert await main(["--repo", "a/b"]) == 1


async def test_a_run_without_a_database_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_MINING_TOKEN", "t")

    assert await main(["--repo", "a/b"]) == 1


@respx.mock
async def test_preflight_names_the_real_cause_of_a_404_on_a_public_repo(
    tmp_path: Path,
) -> None:
    # A fine-grained PAT limited to selected repositories passes the rate-limit check and
    # then 404s on every repo it does not own. At hour three that reads as a bad mining
    # set; here it reads as what it is.
    respx.get(f"{API_ROOT}/repos/celery/celery").mock(httpx.Response(404, json={}))

    async with httpx.AsyncClient() as http:
        client = GitHubClient(http, "t", HttpCache(tmp_path))
        with pytest.raises(GitHubError, match="selected"):
            await preflight(client, "celery/celery")


@respx.mock
async def test_preflight_is_quiet_when_the_token_can_read(tmp_path: Path) -> None:
    respx.get(f"{API_ROOT}/repos/celery/celery").mock(httpx.Response(200, json={"id": 1}))

    async with httpx.AsyncClient() as http:
        await preflight(GitHubClient(http, "t", HttpCache(tmp_path)), "celery/celery")
