"""No em dashes and no en dashes, in code, comments, docs or config.

The rule is stated twice in the project instructions and was checked nowhere until this
test. It scans the working tree rather than the git index, so a new file fails here
before it is ever committed, not after.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Escapes, so this file does not trip its own check.
FORBIDDEN = {"\u2014": "em dash", "\u2013": "en dash"}

SUFFIXES = frozenset(
    {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".sql", ".txt", ".cfg", ".ini"}
)

# Generated, vendored or ignored. uv.lock is written by uv, not by us.
SKIP_DIRECTORIES = frozenset(
    {".git", ".venv", ".cache", ".corpus", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
SKIP_FILES = frozenset({"uv.lock"})


def text_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix in SUFFIXES
        and path.name not in SKIP_FILES
        and not SKIP_DIRECTORIES.intersection(path.relative_to(ROOT).parts)
        and "__pycache__" not in path.parts
    )


def test_the_scan_actually_covers_the_repo() -> None:
    names = {path.relative_to(ROOT).as_posix() for path in text_files()}
    assert "README.md" in names
    assert "src/pr_lens/ingest/mine.py" in names
    assert ".github/workflows/ci.yml" in names


def test_no_file_contains_an_em_or_en_dash() -> None:
    offences = []
    for path in text_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for character, name in FORBIDDEN.items():
                if character in line:
                    offences.append(f"{path.relative_to(ROOT)}:{number}: {name}")
    assert not offences, "use a hyphen, a comma or a full stop instead:\n" + "\n".join(offences)
