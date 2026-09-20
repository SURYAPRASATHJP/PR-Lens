import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pr_lens.jobs.seed import Source, build, compare_url
from pr_lens.review.seeded import TESTBED, seed_source

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SOURCE = Source("pydantic/pydantic-settings", 949, "Add CliVariadicArg", "b" * 40, "h" * 40)


def tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def git(git_dir: Path, *args: str) -> str:
    return subprocess.run(
        [shutil.which("git") or "git", f"--git-dir={git_dir}", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(autouse=True)
def identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Seeder")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "seeder@example.com")


def test_two_commits_carry_exactly_the_reviewed_diff_and_none_of_github(tmp_path: Path) -> None:
    base = tree(
        tmp_path / "base",
        {
            "LICENSE": "MIT",
            "cli.py": "a = 1\n",
            "gone.py": "x\n",
            ".github/workflows/ci.yml": "on: push\n",
        },
    )
    head = tree(
        tmp_path / "head",
        {"LICENSE": "MIT", "cli.py": "a = 2\n", "new.py": "y\n", ".github/dependabot.yml": ""},
    )
    # Force the race rather than wait for it. cli.py is six bytes in both trees, so with
    # one matching mtime git's stat cache calls it unchanged and never stages it. Left to
    # chance this failed about one run in twelve, which is a test that trains you to
    # ignore it.
    same = 1_700_000_000
    for tree_root in (base, head):
        os.utime(tree_root / "cli.py", (same, same))
    base_branch, head_branch = build(tmp_path / "seed.git", SOURCE, base, head)

    changed = git(tmp_path / "seed.git", "diff", "--name-status", base_branch, head_branch)
    assert sorted(changed.split("\n")[:-1]) == ["A\tnew.py", "D\tgone.py", "M\tcli.py"]
    files = git(tmp_path / "seed.git", "ls-tree", "-r", "--name-only", head_branch).split()
    assert "LICENSE" in files
    assert not any(name.startswith(".github") for name in files)
    assert seed_source(TESTBED, head_branch) == ("pydantic/pydantic-settings", 949)
    message = git(tmp_path / "seed.git", "log", "-1", "--format=%B", head_branch)
    assert "pydantic/pydantic-settings#949" in message and "LICENSE" in message


def test_the_compare_link_opens_head_against_base_in_the_testbed() -> None:
    url = compare_url("seed/p__s/949/base", "seed/p__s/949/head", SOURCE)
    assert url.startswith(f"https://github.com/{TESTBED}/compare/seed/p__s/949/base...")
    assert "seed/p__s/949/head?expand=1&title=" in url
