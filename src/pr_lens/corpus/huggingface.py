"""The private dataset repo sink.

Separate from writer.py so that a local run never imports huggingface_hub, and so that
the Hub's commit shape is dealt with in one place. Two limits apply here: under 100 files
per commit, and no dataset viewer on a private repo for a free account, which is why the
ingest job logs shard and row counts. The job log is the only view of what landed.

The corpus is private because free public storage on the Hub became best-effort with no
stated number in 2026, while free private storage is a stated 100 GB.
"""

import json
import logging
import os
import random
import time
from collections.abc import Mapping

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

from pr_lens.corpus.writer import MANIFEST_DIR

logger = logging.getLogger(__name__)

CORPUS_REPO = "suryaprasathjp/pr-lens-corpus"

# The Hub rejects a commit carrying more than 100 files.
MAX_FILES_PER_COMMIT = 100

# Busy, not wrong: rate limited, conflicting with a concurrent commit, or a server error.
RETRYABLE_STATUSES = frozenset({409, 412, 429, 500, 502, 503, 504})
COMMIT_ATTEMPTS = 6


class HuggingFaceSink:
    def __init__(self, repo_id: str = CORPUS_REPO, token: str | None = None) -> None:
        resolved = token or os.environ.get("HF_TOKEN")
        if not resolved:
            # Loudly, and not behind a flag that silently writes nothing. A corpus job
            # that reports success having uploaded nothing is the failure that costs a
            # whole overnight mining run.
            raise RuntimeError(
                "HF_TOKEN is not set, so the corpus cannot be written to "
                f"{repo_id}. Use the local sink for a dry run, or set the fine-grained "
                "token scoped to that dataset repo."
            )
        self.repo_id = repo_id
        self._api = HfApi(token=resolved)

    def read_manifest(self, name: str) -> dict[str, str]:
        path = f"{MANIFEST_DIR}/{name}.json"
        try:
            local = self._api.hf_hub_download(
                repo_id=self.repo_id, repo_type="dataset", filename=path
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            # First ingest of this repo. Everything is new, which is correct.
            return {}
        with open(local, encoding="utf-8") as handle:
            loaded: dict[str, str] = json.load(handle)
        return loaded

    def read(self, name: str) -> bytes | None:
        try:
            local = self._api.hf_hub_download(
                repo_id=self.repo_id, repo_type="dataset", filename=name
            )
        except EntryNotFoundError:
            return None
        with open(local, "rb") as handle:
            return handle.read()

    def write(self, files: Mapping[str, bytes]) -> None:
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=payload)
            for name, payload in sorted(files.items())
        ]
        for start in range(0, len(operations), MAX_FILES_PER_COMMIT):
            batch = operations[start : start + MAX_FILES_PER_COMMIT]
            self._commit(batch)
            logger.info("uploaded %s files to %s", len(batch), self.repo_id)

    def _commit(self, batch: list[CommitOperationAdd]) -> None:
        """One commit, retried when the Hub is busy rather than wrong.

        The Phase 2 embed matrix finishes up to twenty jobs at once against this one repo,
        so a rate limit or a commit landing on a moved branch is expected, and losing an
        hour of embedding to it is not acceptable. Anything else is raised at once.
        """
        for attempt in range(COMMIT_ATTEMPTS):
            try:
                self._api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=batch,
                    commit_message=f"pr-lens: {len(batch)} files",
                )
                return
            except HfHubHTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status not in RETRYABLE_STATUSES or attempt == COMMIT_ATTEMPTS - 1:
                    raise
                wait = min(300.0, 10.0 * 2**attempt) + random.random() * 5  # noqa: S311 -- jitter
                logger.warning("the Hub returned %s, retrying the commit in %.0fs", status, wait)
                time.sleep(wait)
