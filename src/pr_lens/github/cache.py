"""On-disk, content-addressed HTTP cache.

This exists to make a mining run resumable. An Actions job is killed at six hours and the
mining set is 27 repos, so the run that finishes is the second or third one. Replaying
the repos already done has to cost disk reads rather than API calls, or the job spends
its whole rate-limit budget getting back to where it stopped.

The cache is also what makes conditional requests possible. A request that returns 304
does not count against GitHub's primary rate limit, so a stored ETag turns a re-fetch of
unchanged data into a free request rather than a spent one.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    status: int
    etag: str | None
    fetched_at: datetime
    body: bytes
    # A cached page carries no headers, and pagination lives in the Link header. Without
    # this, a replayed run reads page one from disk and then has to go back to the API to
    # find out where page two is, which is the cost the cache exists to avoid.
    next_url: str | None = None

    def age(self, now: datetime | None = None) -> float:
        return ((now or datetime.now(UTC)) - self.fetched_at).total_seconds()


class HttpCache:
    """One metadata file and one body file per request, keyed by a digest of the URL.

    Two files rather than one because bodies are tarballs as often as they are JSON, and
    base64 in a JSON envelope would inflate the biggest thing in the cache by a third.
    The digest is fanned out over 256 directories to stay under the filesystem's
    comfortable directory size, and under the Hub's 10k-entries-per-folder limit if this
    ever moves there.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> CacheEntry | None:
        meta_path, body_path = self._paths(key)
        if not meta_path.exists() or not body_path.exists():
            self.misses += 1
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entry = CacheEntry(
                status=int(meta["status"]),
                etag=meta.get("etag"),
                fetched_at=datetime.fromisoformat(meta["fetched_at"]),
                body=body_path.read_bytes(),
                next_url=meta.get("next_url"),
            )
        except (OSError, ValueError, KeyError):
            # A run killed mid-write leaves a half-written pair. Treat it as absent
            # rather than crashing the resume it was meant to make possible.
            logger.warning("discarding unreadable cache entry for %s", key)
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(
        self,
        key: str,
        *,
        status: int,
        etag: str | None,
        body: bytes,
        next_url: str | None = None,
    ) -> None:
        meta_path, body_path = self._paths(key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        # Body first. A reader checks for both, so an interrupted write is a miss and
        # never a body that does not match its metadata.
        body_path.write_bytes(body)
        meta_path.write_text(
            json.dumps(
                {
                    "key": key,
                    "status": status,
                    "etag": etag,
                    "next_url": next_url,
                    "fetched_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )

    def touch(self, key: str) -> None:
        """Record that a 304 confirmed the stored body is still current."""
        meta_path, _ = self._paths(key)
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["fetched_at"] = datetime.now(UTC).isoformat()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        directory = self.root / digest[:2]
        return directory / f"{digest}.meta.json", directory / f"{digest}.body"
