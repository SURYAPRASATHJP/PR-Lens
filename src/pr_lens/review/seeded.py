"""The testbed's seeded pull requests: one naming rule, written by jobs.seed and read by
jobs.review, kept together so the two cannot drift apart.

A pull request in the testbed whose head branch is seed/<owner>__<name>/<number>/head is a
historical pull request rebuilt for a live run. It retrieves from the repository it was
copied out of and stops below the original number, exactly as replay does. Only the
testbed's branch names carry that meaning: anywhere else they are chosen by whoever opened
the pull request, and must not redirect retrieval to another repository.
"""

import re

TESTBED = "suryaprasathjp/pr-lens-testbed"

_SEED = re.compile(r"^seed/(?P<owner>[\w.-]+)__(?P<name>[\w.-]+)/(?P<number>[1-9]\d*)/head$")


def branches(repo: str, number: int) -> tuple[str, str]:
    """The base and head branch names a seed of repo#number is pushed to."""
    owner, name = repo.split("/", 1)
    stem = f"seed/{owner}__{name}/{number}"
    return f"{stem}/base", f"{stem}/head"


def seed_source(repo: str, head_ref: str) -> tuple[str, int] | None:
    """The repository and pull request a seeded testbed pull request was copied from."""
    if repo.lower() != TESTBED:
        return None
    match = _SEED.match(head_ref)
    if match is None:
        return None
    return f"{match['owner']}/{match['name']}", int(match["number"])
