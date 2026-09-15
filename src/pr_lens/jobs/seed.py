"""Rebuild a historical pull request in the testbed, so the live job can review it for real.

The Phase 4 gate is a comment on a real pull request, and the App is installed on the
testbed alone. So a merged pull request from a tune repository is rebuilt there as two
commits: the merge base, and the commit the human reviewer actually commented on, which is
the code before their review changed it. The live job then reviews exactly what the human
reviewed, and review.seeded.seed_source reads the branch name to retrieve from the source
repository with the same cutoff replay uses, so the human's own comments are never shown.

Run from a machine whose git can push to the testbed over SSH. The branches are new, so
nothing is forced and main is never touched. Everything under .github is left out: the
source repository's workflows and bots are its own automation and would otherwise run on
the testbed. The source's licence file stays, since every tune repository is permissively
licensed and the copy has to carry its notice.
"""

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from pr_lens.eval.split import require_tune
from pr_lens.github.cache import HttpCache
from pr_lens.github.client import IMMUTABLE, GitHubClient
from pr_lens.github.mining import mining_token
from pr_lens.jobs.sandbox import extract
from pr_lens.logging import configure
from pr_lens.review.seeded import TESTBED, branches

logger = logging.getLogger(__name__)

TESTBED_REMOTE = f"git@github.com:{TESTBED}.git"
STRIPPED = (".github",)


@dataclass(frozen=True, slots=True)
class Source:
    repo: str
    number: int
    title: str
    base_sha: str
    head_sha: str


def _git(git_dir: Path, *args: str, work_tree: Path | None = None) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is not installed")
    command = [git, f"--git-dir={git_dir}"]
    if work_tree is not None:
        command.append(f"--work-tree={work_tree}")
    # A fixed program with arguments built here; nothing passes through a shell.
    completed = subprocess.run(  # noqa: S603
        [*command, *args], capture_output=True, text=True, check=True
    )
    return completed.stdout


def build(git_dir: Path, source: Source, base_tree: Path, head_tree: Path) -> tuple[str, str]:
    """Two commits in a fresh repository, base then head, one branch each."""
    for tree in (base_tree, head_tree):
        for name in STRIPPED:
            if (tree / name).exists():
                shutil.rmtree(tree / name)
    base, head = branches(source.repo, source.number)
    _git(git_dir, "init", "--quiet", "--initial-branch", base)
    credit = (
        f"Copied from https://github.com/{source.repo} for a PR-Lens live review test. "
        "The copyright and licence remain the original authors', as the LICENSE file in "
        "this tree states."
    )
    _git(git_dir, "add", "--all", work_tree=base_tree)
    _git(
        git_dir,
        "commit",
        "--quiet",
        "--message",
        f"Base of {source.repo}#{source.number} at {source.base_sha[:12]}",
        "--message",
        credit,
        work_tree=base_tree,
    )
    _git(git_dir, "checkout", "--quiet", "-b", head, work_tree=base_tree)
    _git(git_dir, "add", "--all", work_tree=head_tree)
    _git(
        git_dir,
        "commit",
        "--quiet",
        "--message",
        f"{source.title} ({source.repo}#{source.number} at {source.head_sha[:12]})",
        "--message",
        credit,
        work_tree=head_tree,
    )
    return base, head


async def fetch(repo: str, number: int, head_sha: str | None, into: Path) -> Source:
    """The pull request, its merge base with the reviewed commit, and both trees unpacked."""
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
        client = GitHubClient(http, mining_token() or "", HttpCache(into / "cache"))
        pull = await client.get_json(f"/repos/{repo}/pulls/{number}")
        head = head_sha or str(pull["head"]["sha"])
        compare = await client.get_json(f"/repos/{repo}/compare/{pull['base']['sha']}...{head}")
        base = str(compare["merge_base_commit"]["sha"])
        for sha, name in ((base, "base"), (head, "head")):
            tarball = await client.get_raw(
                f"/repos/{repo}/tarball/{sha}",
                accept="application/vnd.github+json",
                max_age=IMMUTABLE,
            )
            (into / name).mkdir()
            extract(tarball, into / name)
    return Source(repo, number, str(pull["title"]), base, head)


def compare_url(base: str, head: str, source: Source) -> str:
    title = quote(f"[seed] {source.title} ({source.repo}#{source.number})")
    return (
        f"https://github.com/{TESTBED}/compare/{quote(base, safe='/')}..."
        f"{quote(head, safe='/')}?expand=1&title={title}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pr-lens-seed", description=__doc__)
    parser.add_argument("--repo", required=True, help="a tune repository, owner/name")
    parser.add_argument("--pull", type=int, required=True)
    parser.add_argument("--head", help="the commit the human reviewed; default the pull's head")
    parser.add_argument("--remote", default=TESTBED_REMOTE)
    parser.add_argument("--dry-run", action="store_true", help="build the commits, push nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure(os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args(argv)
    require_tune([args.repo])
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        source = asyncio.run(fetch(args.repo, args.pull, args.head, root))
        base, head = build(root / "seed.git", source, root / "base", root / "head")
        stat = _git(root / "seed.git", "diff", "--stat", base, head)
        sys.stdout.write(stat)
        if args.dry_run:
            logger.info("dry run: built %s and %s, pushed nothing", base, head)
            return 0
        _git(root / "seed.git", "push", args.remote, base, head)
    sys.stdout.write(f"\nOpen the pull request:\n{compare_url(base, head, source)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
