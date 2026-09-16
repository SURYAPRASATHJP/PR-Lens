"""Past review comments from the repository under review, nearest to the hunk first.

Serving uses the configuration Phase 2 measured rather than a sibling of it: the same
corpus shards, the same stored gte-modernbert vectors under the same fingerprints, comment
bodies only, exact search pre-filtered to one repository and to review comments. That is
recall@10 0.475 in docs/eval/phase-2-recall-table.md. RRF and the reranker are absent
because the same table found that both made retrieval worse.

Two rules decide what the drafting call must never be shown, and each has a test.

The holdout is refused. Phase 5's golden set is those nine repositories, and drafts over
them feeding the keep-or-kill loop would tune the prompt on the test set.

Automated reviewers are left out. Copilot and its kind write a large share of the review
comments in some repositories, and a draft grounded in "what this project's reviewers care
about" should rest on what its people said. eval.pairs holds the list, and it is the same
one the Phase 2 query set drops.

A pull request never sees its own review comments, or any later pull request's. For
replay they are the answer key: with them in the index a draft can restate what the human
reviewer said and be kept for it. Comments carry no timestamp, only the number of the
pull request they were left on, and numbers are handed out in order, so the cutoff is a
number: strictly below the pull request under review.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pr_lens.eval.corpus import EvalDocument, read_corpus
from pr_lens.eval.pairs import is_automated_login
from pr_lens.eval.split import HOLDOUT, HoldoutViolation
from pr_lens.eval.store import Store
from pr_lens.eval.vectors import corpus_part, document_matrix, load_rows
from pr_lens.jobs.plan import EMBED_PARTS
from pr_lens.retrieval.embed import Encoder, Vectors
from pr_lens.retrieval.search import top_k

# The measured recall is at 10. Five per hunk is what the per-call budget can carry for
# three or four hunks at once, and a comment ranked sixth is rarely the useful one.
COMMENTS_PER_HUNK = 5

# A long review thread message is mostly its first paragraph's point plus discussion.
MAX_COMMENT_CHARS = 600


@dataclass(frozen=True, slots=True)
class PastComment:
    unit_id: str
    path: str | None
    pull_request_number: int
    body: str
    score: float


def refuse_holdout(repo: str) -> None:
    if repo.lower() in {held.lower() for held in HOLDOUT}:
        raise HoldoutViolation(
            f"{repo}: in the Phase 5 holdout. Review never reads, replays or drafts over it."
        )


class CommentIndex:
    def __init__(
        self, repo: str, comments: Sequence[EvalDocument], vectors: Vectors, encoder: Encoder
    ) -> None:
        if len(comments) != vectors.shape[0]:
            raise ValueError("one vector per comment")
        self.repo = repo
        self._comments = tuple(comments)
        self._numbers = np.array([c.pull_request_number for c in comments], dtype=np.int64)
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._encoder = encoder

    @property
    def size(self) -> int:
        return len(self._comments)

    def search(
        self, queries: Sequence[str], *, before: int, k: int = COMMENTS_PER_HUNK
    ) -> list[list[PastComment]]:
        """The k comments nearest each query, from pull requests numbered below `before`."""
        if not queries or not self._comments:
            return [[] for _ in queries]
        allowed = self._numbers < before
        results = []
        for query in self._encoder.encode(queries):
            scores = self._vectors @ query
            results.append(
                [self._past(int(row), float(scores[row])) for row in top_k(scores, k, allowed)]
            )
        return results

    def _past(self, row: int, score: float) -> PastComment:
        comment = self._comments[row]
        body = comment.body or ""
        if len(body) > MAX_COMMENT_CHARS:
            body = body[:MAX_COMMENT_CHARS] + " [cut]"
        return PastComment(comment.unit_id, comment.path, int(self._numbers[row]), body, score)


def load_index(store: Store, repo: str, encoder: Encoder) -> CommentIndex:
    """The repository's review comments, body only, with the vectors the embed job stored."""
    refuse_holdout(repo)
    corpus = read_corpus(store, repo)
    model = encoder.model
    parts = EMBED_PARTS[model.key]
    rows = load_rows(store, [corpus_part(corpus, model, part, parts) for part in range(parts)])
    comments = [
        document
        for document in corpus.documents
        if document.kind == "review_comment"
        and document.pull_request_number is not None
        and document.body
        and not is_automated_login(document.author or "")
    ]
    vectors, _ = document_matrix(rows, comments, body_only=True)
    return CommentIndex(repo, comments, vectors, encoder)
