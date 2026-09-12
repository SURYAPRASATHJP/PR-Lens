"""The tune-side corpus, read back from the shards Phase 1 wrote, as the eval indexes it.

Everything about the index is the corpus unchanged except one thing. A review_comment
unit's text is path:line, then its diff_hunk, then its body, so the unit contains the very
hunk the query is built from. Querying with that hunk retrieves the gold by string match.
So each review comment is also served body only, and the table reports both: the gap
between the two rows is the leakage, measured. The body-only row is the honest number.
"""

import gzip
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pr_lens.corpus.writer import repo_slug
from pr_lens.eval.split import require_tune

Serialisation = Literal["with_hunk", "body_only"]

# Bumped whenever the text an index is built from changes shape, so that vectors embedded
# from the old text are recognised as stale rather than silently reused.
SERIALISATION_VERSION = 1


class ShardStore(Protocol):
    def read_manifest(self, name: str) -> dict[str, str]: ...

    def read(self, name: str) -> bytes | None: ...


@dataclass(frozen=True, slots=True)
class EvalDocument:
    unit_id: str
    repo: str
    kind: str
    path: str | None
    content_hash: str
    text: str
    # Set for review comments only. Every other kind is the same under both
    # serialisations, which is what "everything else about the index is unchanged" means.
    body: str | None = None

    def serialised(self, serialisation: Serialisation) -> str:
        if serialisation == "body_only" and self.body is not None:
            return self.body
        return self.text


@dataclass(frozen=True, slots=True)
class RepoCorpus:
    repo: str
    fingerprint: str
    documents: tuple[EvalDocument, ...]


def load_repo(store: ShardStore, repo: str) -> RepoCorpus:
    """Every unit of one tune repo, in unit id order.

    The fingerprint is a digest of the repo's shard manifest, which Phase 1 already keeps
    as the hash of every shard it wrote. Anything derived from this corpus, vectors
    included, carries it, and is stale the moment the corpus moves.
    """
    require_tune([repo])
    manifest = store.read_manifest(repo_slug(repo))
    if not manifest:
        raise FileNotFoundError(f"no corpus manifest for {repo}. Has it been mined?")
    documents = []
    for shard in sorted(manifest):
        payload = store.read(shard)
        if payload is None:
            raise FileNotFoundError(f"{repo}: the manifest lists {shard} but it is missing")
        for line in gzip.decompress(payload).splitlines():
            documents.append(_document(json.loads(line)))
    documents.sort(key=lambda d: d.unit_id)
    return RepoCorpus(repo=repo, fingerprint=fingerprint(manifest), documents=tuple(documents))


def fingerprint(manifest: dict[str, str]) -> str:
    canonical = json.dumps(
        {"serialisation": SERIALISATION_VERSION, "shards": manifest}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def body_only(record: dict[str, Any]) -> str:
    """The comment body, recovered from the text review_comment_unit built.

    That text is the non-empty parts of (path:line, diff_hunk, body), each stripped and
    joined by a blank line. The header and the hunk are peeled off the front only when
    they are exactly what that function would have written, so a body that happens to
    quote its own hunk further down is left intact.
    """
    text: str = record["text"]
    path = record.get("path")
    if path:
        header, sep, rest = text.partition("\n\n")
        if header.startswith(f"{path}:") and sep:
            text = rest
    hunk = str(record.get("metadata", {}).get("diff_hunk", "")).strip()
    if hunk and text.startswith(hunk + "\n\n"):
        text = text[len(hunk) + 2 :]
    return text


def all_documents(corpora: Iterable[RepoCorpus]) -> list[EvalDocument]:
    return [document for corpus in corpora for document in corpus.documents]


def gold_texts(documents: Sequence[EvalDocument], serialisation: Serialisation) -> dict[str, str]:
    return {d.unit_id: d.serialised(serialisation) for d in documents if d.kind == "review_comment"}


def _document(record: dict[str, Any]) -> EvalDocument:
    kind = str(record["kind"])
    return EvalDocument(
        unit_id=str(record["unit_id"]),
        repo=str(record["repo"]),
        kind=kind,
        path=record.get("path"),
        content_hash=str(record["content_hash"]),
        text=str(record["text"]),
        body=body_only(record) if kind == "review_comment" else None,
    )
