from datetime import UTC, datetime, timedelta
from pathlib import Path

from pr_lens.github.cache import HttpCache


def test_a_stored_response_comes_back_whole(tmp_path: Path) -> None:
    cache = HttpCache(tmp_path)
    cache.put("GET https://api/x", status=200, etag='W/"abc"', body=b"{}", next_url="page2")

    entry = cache.get("GET https://api/x")

    assert entry is not None
    assert entry.status == 200
    assert entry.etag == 'W/"abc"'
    assert entry.body == b"{}"
    assert entry.next_url == "page2"
    assert cache.hits == 1


def test_a_missing_key_is_a_miss_not_an_error(tmp_path: Path) -> None:
    assert HttpCache(tmp_path).get("GET https://api/never") is None


def test_binary_bodies_survive_the_round_trip(tmp_path: Path) -> None:
    # Tarballs go through here, and they are the largest thing in the cache.
    cache = HttpCache(tmp_path)
    payload = bytes(range(256)) * 100
    cache.put("GET tarball", status=200, etag=None, body=payload)
    entry = cache.get("GET tarball")
    assert entry is not None
    assert entry.body == payload


def test_a_run_killed_mid_write_reads_as_a_miss(tmp_path: Path) -> None:
    cache = HttpCache(tmp_path)
    cache.put("GET https://api/x", status=200, etag=None, body=b"{}")
    next(tmp_path.rglob("*.meta.json")).write_text("{ truncated")

    assert cache.get("GET https://api/x") is None


def test_a_body_without_its_metadata_reads_as_a_miss(tmp_path: Path) -> None:
    cache = HttpCache(tmp_path)
    cache.put("GET https://api/x", status=200, etag=None, body=b"{}")
    next(tmp_path.rglob("*.meta.json")).unlink()

    assert cache.get("GET https://api/x") is None


def test_touch_resets_the_age_a_304_just_confirmed(tmp_path: Path) -> None:
    cache = HttpCache(tmp_path)
    cache.put("GET https://api/x", status=200, etag='W/"a"', body=b"{}")
    meta = next(tmp_path.rglob("*.meta.json"))
    stale = datetime.now(UTC) - timedelta(days=2)
    meta.write_text(meta.read_text().replace(_fetched_at(meta), stale.isoformat()))
    assert cache.get("GET https://api/x").age() > 86400  # type: ignore[union-attr]

    cache.touch("GET https://api/x")

    entry = cache.get("GET https://api/x")
    assert entry is not None
    assert entry.age() < 5
    assert entry.etag == 'W/"a"'


def test_keys_are_fanned_out_so_no_directory_grows_without_limit(tmp_path: Path) -> None:
    cache = HttpCache(tmp_path)
    for index in range(50):
        cache.put(f"GET https://api/{index}", status=200, etag=None, body=b"{}")

    assert len(list(tmp_path.iterdir())) > 1


def _fetched_at(meta: Path) -> str:
    import json

    return str(json.loads(meta.read_text())["fetched_at"])
