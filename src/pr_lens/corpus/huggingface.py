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
from collections.abc import Mapping

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from pr_lens.corpus.writer import MANIFEST_DIR

logger = logging.getLogger(__name__)

CORPUS_REPO = "suryaprasathjp/pr-lens-corpus"

# The Hub rejects a commit carrying more than 100 files.
MAX_FILES_PER_COMMIT = 100


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

    def write(self, files: Mapping[str, bytes]) -> None:
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=payload)
            for name, payload in sorted(files.items())
        ]
        for start in range(0, len(operations), MAX_FILES_PER_COMMIT):
            batch = operations[start : start + MAX_FILES_PER_COMMIT]
            self._api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=batch,
                commit_message=f"ingest: {len(batch)} shards",
            )
            logger.info("uploaded %s files to %s", len(batch), self.repo_id)
