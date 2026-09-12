"""Which repos Phase 2 may tune against, and which it must never see.

Split by repo and never by pull request. Splitting by PR leaks: two PRs from one repo
share reviewers, conventions and code, so a model tuned on one has partly seen the other.
And the split has to exist before anything is embedded, because a Phase 2 retriever tuned
against repos that later turn up in the Phase 5 held-out set inflates every Phase 5
number by an amount nobody can recover.

Assigned on 12 Sep 2026 by ranking the 27 mining repos on review-comment velocity and
taking every third one for the holdout, so neither side is all high-traffic repos. Both
NOASSERTION-licence watch items, celery and pymc, landed on the tune side, so a licence
review that drops them cannot disturb the holdout.

This is the only place the split is written down. Phase 5 imports it too.
"""

from collections.abc import Iterable

TUNE: frozenset[str] = frozenset(
    {
        "celery/celery",
        "aio-libs/aiohttp",
        "sqlfluff/sqlfluff",
        "pytest-dev/pytest",
        "litestar-org/litestar",
        "ibis-project/ibis",
        "pymc-devs/pymc",
        "strawberry-graphql/strawberry",
        "python-poetry/poetry",
        "unionai-oss/pandera",
        "pypa/hatch",
        "pydantic/pydantic-settings",
        "joblib/joblib",
        "fastapi/typer",
        "Textualize/textual",
        "tox-dev/tox",
        "python-attrs/attrs",
        "encode/httpx",
    }
)

HOLDOUT: frozenset[str] = frozenset(
    {
        "redis/redis-py",
        "jazzband/pip-tools",
        "scrapy/scrapy",
        "fsspec/filesystem_spec",
        "pallets/click",
        "urllib3/urllib3",
        "fastapi/sqlmodel",
        "pdm-project/pdm",
        "marshmallow-code/marshmallow",
    }
)

MINING_SET: frozenset[str] = TUNE | HOLDOUT


class HoldoutViolation(ValueError):
    """Something tried to put a held-out or unknown repo in front of the Phase 2 harness."""


def require_tune(repos: Iterable[str]) -> list[str]:
    """Every Phase 2 entry point passes its repo list through here before doing any work.

    Comparison is case-insensitive because GitHub is: Textualize/textual and
    textualize/textual are the same repo, and a lowercased MINING_REPOS variable must not
    slip a holdout repo past a case-sensitive check.
    """
    tune = {repo.lower() for repo in TUNE}
    holdout = {repo.lower() for repo in HOLDOUT}
    requested = list(dict.fromkeys(repos))
    leaked = sorted(repo for repo in requested if repo.lower() in holdout)
    if leaked:
        raise HoldoutViolation(
            f"{', '.join(leaked)}: in the Phase 5 holdout. Phase 2 never embeds, queries "
            "or reads a held-out repo."
        )
    unknown = sorted(repo for repo in requested if repo.lower() not in tune)
    if unknown:
        raise HoldoutViolation(
            f"{', '.join(unknown)}: not in the tune split. Only the 18 tune repos are "
            "eligible for Phase 2."
        )
    return requested


def tune_repos() -> list[str]:
    """The tune split in a stable order, which is what every Phase 2 job iterates."""
    return sorted(TUNE, key=str.lower)
