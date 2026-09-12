"""Where Phase 2 reads the corpus from and writes what it derives.

The private dataset repo in Actions, a local directory otherwise. Nothing here ever goes
to Neon: the tune corpus as vectors is several hundred megabytes against a 0.5 GB
database and a 5 GB monthly egress cap, and none of it is served live.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from pr_lens.corpus.writer import LocalSink


class Store(Protocol):
    def read_manifest(self, name: str) -> dict[str, str]: ...

    def read(self, name: str) -> bytes | None: ...

    def write(self, files: Mapping[str, bytes]) -> None: ...


def build_store(sink: Literal["local", "huggingface"] | str, root: str | Path) -> Store:
    if sink == "local":
        return LocalSink(Path(root))
    from pr_lens.corpus.huggingface import HuggingFaceSink

    return HuggingFaceSink()
