"""The cross-encoder second stage.

Same family, tokenizer and licence as the 8k embedder. A cross-encoder reads the query and
the candidate together, so it can see what a bi-encoder squeezes into one vector per side,
but it costs a forward pass per pair. That cost is why the plateau matters: the number of
candidates past which reranking stops buying recall is the serving configuration.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

RERANKER_NAME = "Alibaba-NLP/gte-reranker-modernbert-base"
RERANKER_REVISION = "f7481e6055501a30fb19d090657df9ec1f79ab2c"


class CrossEncoderReranker:
    def __init__(self, max_tokens: int, batch_size: int = 8) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.max_tokens = max_tokens
        self._batch_size = batch_size
        # float32 forced, for the reason given in embed.SentenceEncoder: the checkpoint is
        # stored in float16, and float16 on a CPU is emulated and many times slower.
        self._model = CrossEncoder(
            RERANKER_NAME,
            revision=RERANKER_REVISION,
            device="cpu",
            max_length=max_tokens,
            model_kwargs={"dtype": torch.float32},
        )

    def score(self, query: str, documents: Sequence[str]) -> npt.NDArray[np.float32]:
        if not documents:
            return np.zeros(0, dtype=np.float32)
        scores = self._model.predict(
            [(query, document) for document in documents],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32)
