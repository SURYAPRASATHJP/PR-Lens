"""Sharded, deterministic corpus writing.

The Hub's shape limits decide the layout: under 100k files per repo, under 10k entries
per folder, under 200 GB per file, under 100 files per commit. A file per chunk would hit
the first of those inside the 27-repo mining set, so units are packed into one gzipped
JSONL shard per repo per kind, rolling to a second shard by size.

Deterministic because idempotence has to hold at the storage layer too, not only in
Postgres. Units are sorted by id, keys are sorted, and gzip is told to write an mtime of
zero, so identical input produces a byte-identical shard. A manifest of shard hashes then
lets an unchanged shard be skipped rather than re-uploaded, which is the difference
between a no-op re-ingest and one that rewrites the whole corpus every night.
"""

import gzip
import hashlib
import io
import json
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pr_lens.ingest.units import CorpusUnit

logger = logging.getLogger(__name__)

MAX_SHARD_BYTES = 32 * 1024 * 1024
MAX_SHARD_ROWS = 20_000

MANIFEST_DIR = "manifests"


class Sink(Protocol):
    """Where shards land. Local disk by default, the private dataset repo in Actions."""

    def read_manifest(self, name: str) -> dict[str, str]: ...

    def write(self, files: Mapping[str, bytes]) -> None: ...


@dataclass(frozen=True, slots=True)
class ShardReport:
    repo: str
    rows: int
    written: tuple[str, ...] = field(default_factory=tuple)
    unchanged: tuple[str, ...] = field(default_factory=tuple)
    # unit id to the shard holding its text. Postgres stores this so a retrieval hit can
    # be turned back into the text without scanning the dataset repo.
    placement: Mapping[str, str] = field(default_factory=dict)

    @property
    def shards(self) -> int:
        return len(self.written) + len(self.unchanged)


class LocalSink:
    def __init__(self, root: Path) -> None:
        self.root = root

    def read_manifest(self, name: str) -> dict[str, str]:
        path = self.root / MANIFEST_DIR / f"{name}.json"
        if not path.exists():
            return {}
        loaded: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def write(self, files: Mapping[str, bytes]) -> None:
        for name, payload in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Via a temporary file and a rename, so two writers racing on one path leave
            # one whole file rather than an interleaving of both.
            partial = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
            partial.write_bytes(payload)
            partial.replace(path)

    def read(self, name: str) -> bytes | None:
        path = self.root / name
        return path.read_bytes() if path.exists() else None


def write_units(sink: Sink, repo: str, units: Iterable[CorpusUnit]) -> ShardReport:
    """Pack units into shards and send only the ones whose contents actually changed."""
    slug = repo_slug(repo)
    shards, placement = _build_shards(slug, units)
    previous = sink.read_manifest(slug)

    manifest = {name: _sha256(payload) for name, payload in shards.items()}
    changed = {
        name: payload for name, payload in shards.items() if previous.get(name) != manifest[name]
    }
    unchanged = tuple(sorted(set(shards) - set(changed)))

    # The manifest also records shards that no longer exist, so a repo that lost a file
    # does not leave a stale entry claiming the old shard is still current.
    if changed or set(previous) != set(manifest):
        changed[f"{MANIFEST_DIR}/{slug}.json"] = _canonical_json(manifest)

    if changed:
        sink.write(changed)

    rows = sum(payload.count(b"\n") for payload in _decompressed(shards).values())
    report = ShardReport(
        repo=repo,
        rows=rows,
        written=tuple(sorted(name for name in changed if not name.startswith(MANIFEST_DIR))),
        unchanged=unchanged,
        placement=placement,
    )
    logger.info(
        "%s: %s rows across %s shards, %s written, %s unchanged",
        repo,
        report.rows,
        report.shards,
        len(report.written),
        len(report.unchanged),
    )
    return report


def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


def shard_name(slug: str, kind: str, index: int) -> str:
    return f"{kind}/{slug}-{index:05d}.jsonl.gz"


def _build_shards(
    slug: str, units: Iterable[CorpusUnit]
) -> tuple[dict[str, bytes], dict[str, str]]:
    by_kind: dict[str, list[CorpusUnit]] = defaultdict(list)
    for unit in units:
        by_kind[unit.kind].append(unit)

    shards: dict[str, bytes] = {}
    placement: dict[str, str] = {}
    for kind, kind_units in sorted(by_kind.items()):
        buffer: list[bytes] = []
        size = 0
        index = 0
        for unit in sorted(kind_units, key=lambda u: u.unit_id):
            line = _canonical_json(unit.as_record()) + b"\n"
            if buffer and (size + len(line) > MAX_SHARD_BYTES or len(buffer) >= MAX_SHARD_ROWS):
                shards[shard_name(slug, kind, index)] = _gzip(b"".join(buffer))
                buffer, size, index = [], 0, index + 1
            buffer.append(line)
            size += len(line)
            placement[unit.unit_id] = shard_name(slug, kind, index)
        if buffer:
            shards[shard_name(slug, kind, index)] = _gzip(b"".join(buffer))
    return shards, placement


def _decompressed(shards: Mapping[str, bytes]) -> dict[str, bytes]:
    return {name: gzip.decompress(payload) for name, payload in shards.items()}


def _gzip(payload: bytes) -> bytes:
    """mtime=0 is the whole point: gzip otherwise stamps the clock into every shard."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
