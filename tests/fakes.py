import hashlib
import re
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pr_lens.retrieval.embed import EmbeddingModel


class BagEncoder:
    """Hashed bag of words, so a query and a document sharing words land close together."""

    def __init__(self, model: EmbeddingModel, batch_size: int = 16) -> None:
        del batch_size
        self.model = model

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        rows = np.zeros((len(texts), self.model.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.model.dims
                rows[i, slot] += 1.0
            norm = np.linalg.norm(rows[i])
            rows[i] = rows[i] / norm if norm else rows[i]
            if not norm:
                rows[i, 0] = 1.0
        return rows

    def token_counts(self, texts: Sequence[str]) -> npt.NDArray[np.int32]:
        return np.asarray([len(t.split()) * 3 for t in texts], dtype=np.int32)
