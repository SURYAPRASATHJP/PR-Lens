"""The one thing every ingest path produces, and the hash the idempotence gate rests on."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

UnitKind = Literal["source", "doc", "pull_request", "review_comment", "issue"]


@dataclass(frozen=True, slots=True)
class CorpusUnit:
    """One retrievable thing: a function, a docs section, a PR body, a review comment.

    identity is the part of the unit that does not change when its content does. It is
    what unit_id is derived from, so a function whose body is edited updates a row rather
    than adding one. Get this wrong in the direction of including a commit sha and
    re-ingest never converges, which is exactly what the gate would catch.
    """

    repo: str
    kind: UnitKind
    identity: str
    text: str
    ref: str | None = None
    path: str | None = None
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def unit_id(self) -> str:
        return _digest(f"{self.repo}\x00{self.kind}\x00{self.identity}")

    @property
    def content_hash(self) -> str:
        """Changes when and only when the shard record changes.

        That equivalence is the point. The corpus writer skips uploading a shard whose
        bytes match the manifest, and that is only safe if the hash covers everything the
        record contains. So the hash is taken over the record itself, minus the hash.

        Line numbers are in there because a chunk that slid down a file is genuinely a
        different retrieval result, and leaving them out would mean citing stale lines
        forever. ref is not, because it changes on every unrelated push, and a corpus
        that rewrote itself on every push would spend the whole Hub budget saying nothing.
        """
        return _digest(_canonical(self._content()))

    @property
    def char_count(self) -> int:
        return len(self.text)

    def as_row(self) -> dict[str, Any]:
        """The Postgres index. No text: Neon free is 0.5 GB and the text is elsewhere."""
        return {**self._indexed_fields(), "content_hash": self.content_hash, "ref": self.ref}

    def as_record(self) -> dict[str, Any]:
        """The dataset-repo record: the content, plus enough identity to find it again.

        ref is deliberately absent. It is the one field that moves without the content
        moving, and a record carrying it would make every shard differ after any push.
        The commit a unit was last seen at is index metadata, so it lives in Postgres.
        """
        return {**self._content(), "content_hash": self.content_hash}

    def _content(self) -> dict[str, Any]:
        """The shard record without its hash, which is exactly what the hash covers."""
        return {**self._indexed_fields(), "text": self.text}

    def _indexed_fields(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "repo": self.repo,
            "kind": self.kind,
            "path": self.path,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
