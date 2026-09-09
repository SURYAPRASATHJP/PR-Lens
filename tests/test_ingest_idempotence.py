"""The Phase 1 gate: re-running ingest writes zero new rows.

Not few. Zero. The whole pipeline runs here against a real Postgres and a mocked GitHub,
rather than testing the upsert on its own, because every layer between the API and the
row can break idempotence: a chunker that is not deterministic, a hash that includes the
commit sha, a shard name that carries a timestamp. Testing the last step alone would pass
while the pipeline as a whole rewrote the corpus every night.
"""

import gzip
import io
import json
import tarfile
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

import asyncpg
import httpx
import pytest
import respx

from pr_lens.corpus.writer import LocalSink
from pr_lens.db.connection import connect
from pr_lens.db.migrate import migrate
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import API_ROOT, GitHubClient
from pr_lens.ingest.mine import MiningLimits
from pr_lens.ingest.pipeline import ingest_repo

from .conftest import database_url, requires_postgres

pytestmark = requires_postgres

REPO = "octocat/hello-world"
HEAD = "c" * 40

TREE = {
    "src/parser.py": (
        '"""Parse things."""\n\nimport os\n\n\n'
        "def parse(value):\n"
        '    """Parse a value."""\n'
        "    return value + 1\n\n\n"
        "class Walker:\n"
        "    def walk(self, node):\n"
        "        return node\n"
    ),
    "README.md": (
        "# Hello World\n\nAn introduction long enough to stand on its own as a chunk.\n\n"
        "## Install\n\nRun the installer and then run it a second time for luck.\n"
    ),
    "node_modules/left-pad/index.js": "module.exports = 1;\n",
    "assets/logo.png": "\x00\x01binary\x02",
}


@pytest.fixture
async def conn() -> AsyncIterator[asyncpg.Connection]:
    dsn = database_url()
    assert dsn is not None
    await migrate(dsn)
    connection = await connect(dsn)
    await connection.execute("truncate corpus_units, ingest_runs, corpus_embeddings")
    yield connection
    await connection.close()


@pytest.fixture
async def clients(tmp_path: Path) -> AsyncIterator[Callable[[], GitHubClient]]:
    """A factory, because a cold cache is what a run on a later day actually has.

    Sharing one cache models re-running within the six-hour listing window, where the
    head sha is deliberately read from disk. Handing out a fresh one models tomorrow.
    """
    counter = count()
    async with httpx.AsyncClient(follow_redirects=True) as http:
        yield lambda: GitHubClient(
            http, "test-token", HttpCache(tmp_path / f"cache-{next(counter)}")
        )


@pytest.fixture
def client(clients: Callable[[], GitHubClient]) -> GitHubClient:
    return clients()


@pytest.fixture
def sink(tmp_path: Path) -> LocalSink:
    return LocalSink(tmp_path / "corpus")


@respx.mock
async def test_a_second_ingest_writes_nothing_at_all(
    conn: asyncpg.Connection, client: GitHubClient, sink: LocalSink
) -> None:
    fake_github()

    first = await ingest_repo(client, conn, sink, REPO, MiningLimits())
    second = await ingest_repo(client, conn, sink, REPO, MiningLimits())

    assert first.counts.inserted > 0
    assert second.counts.inserted == 0
    assert second.counts.updated == 0
    assert second.counts.unchanged == first.counts.seen
    assert second.pruned == 0
    assert second.shards_written == 0
    assert second.wrote_nothing


@respx.mock
async def test_a_cold_cache_re_ingest_of_the_same_data_still_writes_nothing(
    conn: asyncpg.Connection, clients: Callable[[], GitHubClient], sink: LocalSink
) -> None:
    # The stronger version of the gate. Nothing is carried over between the two runs
    # except Postgres and the shard manifest, so convergence cannot be an artefact of a
    # cache hit somewhere in the middle.
    fake_github()
    await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    second = await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    assert second.wrote_nothing


@respx.mock
async def test_a_new_commit_that_changed_nothing_still_writes_nothing(
    conn: asyncpg.Connection, clients: Callable[[], GitHubClient], sink: LocalSink
) -> None:
    # The realistic case. A repo gets a push that touches one file we do not index, and
    # HEAD moves. If the sha reached the hash, this alone would rewrite the whole corpus.
    fake_github()
    await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    respx.reset()
    fake_github(sha="d" * 40)
    second = await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    assert second.ref == "d" * 40
    assert second.wrote_nothing


@respx.mock
async def test_an_edited_function_updates_one_row_and_inserts_none(
    conn: asyncpg.Connection, clients: Callable[[], GitHubClient], sink: LocalSink
) -> None:
    fake_github()
    await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    edited = dict(TREE)
    edited["src/parser.py"] = TREE["src/parser.py"].replace("return value + 1", "return value + 2")
    respx.reset()
    fake_github(sha="d" * 40, tree=edited)
    second = await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    assert (second.counts.inserted, second.counts.updated) == (0, 1)
    row = await conn.fetchrow("select * from corpus_units where symbol = 'parse'")
    assert row is not None
    assert row["ref"] == "d" * 40
    assert row["updated_at"] > row["first_seen_at"]


@respx.mock
async def test_a_deleted_file_is_pruned_rather_than_left_indexed_forever(
    conn: asyncpg.Connection, clients: Callable[[], GitHubClient], sink: LocalSink
) -> None:
    fake_github()
    await ingest_repo(clients(), conn, sink, REPO, MiningLimits())
    before = await conn.fetchval("select count(*) from corpus_units where path = 'README.md'")
    assert before > 0

    remaining = {k: v for k, v in TREE.items() if k != "README.md"}
    respx.reset()
    fake_github(sha="d" * 40, tree=remaining)
    second = await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    assert second.pruned == before
    assert await conn.fetchval("select count(*) from corpus_units where path = 'README.md'") == 0


@respx.mock
async def test_a_pull_request_that_fell_out_of_the_window_is_not_pruned(
    conn: asyncpg.Connection, clients: Callable[[], GitHubClient], sink: LocalSink
) -> None:
    # The API-backed kinds are a window over the most recent N, so absence means "older
    # than the window" rather than "deleted". Pruning them would eat the corpus.
    fake_github()
    await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    respx.reset()
    fake_github(sha="d" * 40, pulls=[], comments=[], issues=[])
    await ingest_repo(clients(), conn, sink, REPO, MiningLimits())

    assert await conn.fetchval("select count(*) from corpus_units where kind = 'pull_request'") == 1


@respx.mock
async def test_the_indexed_units_are_the_ones_we_meant_to_index(
    conn: asyncpg.Connection, client: GitHubClient, sink: LocalSink
) -> None:
    fake_github()

    await ingest_repo(client, conn, sink, REPO, MiningLimits())

    rows = await conn.fetch("select kind, count(*) as n from corpus_units group by kind")
    kinds = {row["kind"]: row["n"] for row in rows}
    assert kinds["pull_request"] == 1
    assert kinds["review_comment"] == 1
    assert kinds["issue"] == 1
    assert kinds["source"] >= 2
    assert kinds["doc"] >= 1
    paths = {r["path"] for r in await conn.fetch("select distinct path from corpus_units")}
    assert "node_modules/left-pad/index.js" not in paths
    assert "assets/logo.png" not in paths


@respx.mock
async def test_a_closed_but_unmerged_pull_request_is_not_indexed(
    conn: asyncpg.Connection, client: GitHubClient, sink: LocalSink
) -> None:
    fake_github(pulls=[_pull(1, merged=True), _pull(2, merged=False)])

    await ingest_repo(client, conn, sink, REPO, MiningLimits())

    numbers = await conn.fetch(
        "select (metadata->>'number')::int as n from corpus_units where kind = 'pull_request'"
    )
    assert [r["n"] for r in numbers] == [1]


@respx.mock
async def test_a_review_comment_carries_its_hunk_so_it_retrieves_on_code(
    conn: asyncpg.Connection, client: GitHubClient, sink: LocalSink
) -> None:
    fake_github()

    await ingest_repo(client, conn, sink, REPO, MiningLimits())

    row = await conn.fetchrow("select * from corpus_units where kind = 'review_comment'")
    assert row is not None
    assert row["path"] == "src/parser.py"
    shard = json.loads(row["metadata"])["diff_hunk"]
    assert "@@" in shard
    text = _text_of(sink, row["shard"], row["unit_id"])
    assert "@@" in text
    assert "This shadows the builtin." in text


@respx.mock
async def test_every_row_points_at_a_shard_that_holds_its_text(
    conn: asyncpg.Connection, client: GitHubClient, sink: LocalSink
) -> None:
    fake_github()

    await ingest_repo(client, conn, sink, REPO, MiningLimits())

    for row in await conn.fetch("select unit_id, shard, char_count from corpus_units"):
        text = _text_of(sink, row["shard"], row["unit_id"])
        assert len(text) == row["char_count"]


@respx.mock
async def test_the_run_is_logged_because_a_private_dataset_repo_has_no_viewer(
    conn: asyncpg.Connection, client: GitHubClient, sink: LocalSink
) -> None:
    fake_github()

    report = await ingest_repo(client, conn, sink, REPO, MiningLimits())

    run = await conn.fetchrow("select * from ingest_runs order by run_id desc limit 1")
    assert run is not None
    assert run["repo"] == REPO
    assert run["ref"] == HEAD
    assert run["units_seen"] == report.counts.seen
    assert run["inserted"] == report.counts.inserted
    assert run["shards"] == report.shards
    assert run["finished_at"] is not None
    assert run["error"] is None


@respx.mock
async def test_a_failed_run_is_recorded_with_its_error(
    conn: asyncpg.Connection, client: GitHubClient, sink: LocalSink
) -> None:
    fake_github()
    respx.get(f"{API_ROOT}/repos/{REPO}/pulls").mock(httpx.Response(500, text="boom"))

    with pytest.raises(Exception, match="500"):
        await ingest_repo(client, conn, sink, REPO, MiningLimits())

    run = await conn.fetchrow("select * from ingest_runs order by run_id desc limit 1")
    assert run is not None
    assert run["error"] is not None
    assert run["finished_at"] is not None


def fake_github(
    *,
    sha: str = HEAD,
    tree: dict[str, str] | None = None,
    pulls: list[dict[str, object]] | None = None,
    comments: list[dict[str, object]] | None = None,
    issues: list[dict[str, object]] | None = None,
) -> None:
    respx.get(f"{API_ROOT}/repos/{REPO}").mock(httpx.Response(200, json={"default_branch": "main"}))
    respx.get(f"{API_ROOT}/repos/{REPO}/commits/main").mock(httpx.Response(200, json={"sha": sha}))
    respx.get(f"{API_ROOT}/repos/{REPO}/tarball/{sha}").mock(
        httpx.Response(200, content=_tarball(tree if tree is not None else TREE))
    )
    respx.get(f"{API_ROOT}/repos/{REPO}/pulls").mock(
        httpx.Response(200, json=pulls if pulls is not None else [_pull(1, merged=True)])
    )
    respx.get(f"{API_ROOT}/repos/{REPO}/pulls/comments").mock(
        httpx.Response(200, json=comments if comments is not None else [_comment()])
    )
    respx.get(f"{API_ROOT}/repos/{REPO}/issues").mock(
        httpx.Response(200, json=issues if issues is not None else [_issue()])
    )


def _pull(number: int, *, merged: bool) -> dict[str, object]:
    return {
        "number": number,
        "title": f"Tidy the parser, take {number}",
        "body": "Renames the loop variable and adds a test for the empty case.",
        "merged_at": "2026-09-01T00:00:00Z" if merged else None,
        "merge_commit_sha": "e" * 40 if merged else None,
        "user": {"login": "reviewer"},
        "html_url": f"https://github.com/{REPO}/pull/{number}",
    }


def _comment() -> dict[str, object]:
    return {
        "id": 900100,
        "body": "This shadows the builtin. Rename it before someone imports this module.",
        "path": "src/parser.py",
        "line": 7,
        "side": "RIGHT",
        "commit_id": "f" * 40,
        "diff_hunk": "@@ -5,3 +5,4 @@ def parse(value):\n     context\n+    id = value\n",
        "user": {"login": "reviewer"},
        "html_url": f"https://github.com/{REPO}/pull/1#discussion_r900100",
        "pull_request_url": f"{API_ROOT}/repos/{REPO}/pulls/1",
    }


def _issue() -> dict[str, object]:
    return {
        "number": 42,
        "title": "Parser drops the last token",
        "body": "Reproduced on 3.12. The final token is silently discarded when input ends.",
        "state": "open",
        "user": {"login": "reporter"},
        "html_url": f"https://github.com/{REPO}/issues/42",
    }


def _tarball(tree: dict[str, str]) -> bytes:
    """A GitHub tarball, prefix directory and all."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content in sorted(tree.items()):
            payload = content.encode("utf-8", errors="surrogateescape")
            info = tarfile.TarInfo(f"octocat-hello-world-abc1234/{path}")
            info.size = len(payload)
            info.mtime = int(datetime(2026, 9, 9, tzinfo=UTC).timestamp())
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _text_of(sink: LocalSink, shard: str, unit_id: str) -> str:
    lines = gzip.decompress((sink.root / shard).read_bytes()).splitlines()
    for line in lines:
        record = json.loads(line)
        if record["unit_id"] == unit_id:
            return str(record["text"])
    raise AssertionError(f"{unit_id} is not in {shard}")
