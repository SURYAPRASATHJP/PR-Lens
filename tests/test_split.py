import pytest

from pr_lens.eval.split import HOLDOUT, MINING_SET, TUNE, HoldoutViolation, require_tune, tune_repos


def test_the_two_sides_are_disjoint() -> None:
    assert not {r.lower() for r in TUNE} & {r.lower() for r in HOLDOUT}


def test_the_split_covers_the_whole_mining_set() -> None:
    assert len(TUNE) == 18
    assert len(HOLDOUT) == 9
    assert len(MINING_SET) == 27
    assert len({r.lower() for r in MINING_SET}) == 27


def test_a_holdout_repo_is_refused() -> None:
    with pytest.raises(HoldoutViolation, match="holdout"):
        require_tune(["celery/celery", "pallets/click"])


def test_the_holdout_check_ignores_case() -> None:
    with pytest.raises(HoldoutViolation, match="holdout"):
        require_tune(["Pallets/Click"])


def test_a_repo_outside_the_mining_set_is_refused() -> None:
    with pytest.raises(HoldoutViolation, match="not in the tune split"):
        require_tune(["octocat/hello-world"])


def test_tune_repos_passes_its_own_check() -> None:
    assert require_tune(tune_repos()) == tune_repos()
    assert set(tune_repos()) == TUNE
