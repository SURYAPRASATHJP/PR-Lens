"""Find how a repository runs its tests, or say plainly that it does not.

Python and JavaScript or TypeScript only, decided 14 Sep 2026. Everything else is NO_TESTS
with a reason naming what was found, which is a measured outcome and not a failure.

Every answer carries its reason, and every ambiguity is resolved by a fixed rule that is
written into the notes. Nothing here guesses silently, because a Phase 4 comment that
says "your tests fail" has to be traceable to why this module chose the command it did.

Only files are read. Nothing in the repository is imported or executed here, which is
why Python dependencies come from static metadata alone: pyproject.toml, setup.cfg,
requirements files and Poetry's tables. A setup.py is never run to ask it what it needs.
A repository whose dependencies only setup.py knows gets the ones it also wrote down
somewhere static, and its tests report the rest as import errors, which is honest.
"""

import configparser
import json
import os
import re
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name

from pr_lens.sandbox.images import NODE_IMAGE, PYTHON_IMAGE
from pr_lens.sandbox.runner import PROJECT_SITE
from pr_lens.sandbox.spec import MAX_REQUIREMENTS, Fetcher, SpecError, check_requirement

# Dependency tables whose names mean "what the tests need". Checked in this order, and dev
# only when none of the others exists, because dev usually means linters and docs
# tooling as well, and each extra package is another chance of an sdist-only dependency
# failing the binary-only fetch.
TEST_KEYS = ("test", "tests", "testing")
FALLBACK_KEYS = ("dev",)

# Checked in this order, each one that exists is read.
REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-test.txt",
    "requirements_test.txt",
    "requirements-tests.txt",
    "test-requirements.txt",
    "requirements-dev.txt",
    "requirements_dev.txt",
    "dev-requirements.txt",
    "requirements/test.txt",
    "requirements/tests.txt",
    "requirements/testing.txt",
    "requirements/dev.txt",
)

# The source is a tarball with no .git, and version plugins that read git history fail
# the project build without one of these. The version is a placeholder, not a claim.
VERSION_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0"),
    ("PDM_BUILD_SCM_VERSION", "0.0.0"),
    ("POETRY_DYNAMIC_VERSIONING_BYPASS", "0.0.0"),
)

# Builds the project against the fetched dependencies, with no network, into a directory
# on PYTHONPATH. Not editable, the way tox installs, so generated files such as a
# setuptools_scm _version.py exist and importlib.metadata can find the distribution.
PYTHON_SETUP = (
    "python",
    "-m",
    "pip",
    "install",
    "--no-deps",
    "--no-build-isolation",
    "--no-index",
    "--no-cache-dir",
    "--disable-pip-version-check",
    "--quiet",
    "--target",
    PROJECT_SITE,
    ".",
)

# -rfE lists every failure and error in the short summary, which is what evidence.py
# reads. Colour off, because the output is parsed, not looked at.
PYTEST_COMMAND = ("python", "-m", "pytest", "-rfE", "--tb=short", "--color=no")

# pytest's own exit code for "collected nothing". That is NO_TESTS, not a failure.
PYTEST_NO_TESTS_EXIT = 5

# npm init writes this as the default test script. It is a placeholder, not a suite.
_NPM_PLACEHOLDER = re.compile(r"no test specified")

# A walk is bounded so a pathological repository costs a bounded amount to look at.
MAX_FILES_SCANNED = 50_000
MAX_MANIFEST_BYTES = 1_000_000

_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".nox",
        "__pycache__",
        "site-packages",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
    }
)

_PY_TEST_FILE = re.compile(r"^(test_.*|.*_test)\.py$")
_JS_TEST_FILE = re.compile(r"^.*\.(test|spec)\.[cm]?[jt]sx?$")

# Marker files for languages this phase does not run, so the reason can name one.
_OTHER_LANGUAGES = (
    ("go.mod", "Go"),
    ("Cargo.toml", "Rust"),
    ("pom.xml", "Java"),
    ("build.gradle", "Java or Kotlin"),
    ("build.gradle.kts", "Kotlin"),
    ("Gemfile", "Ruby"),
    ("composer.json", "PHP"),
    ("mix.exs", "Elixir"),
    ("Package.swift", "Swift"),
)


class Language(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class Framework(StrEnum):
    """Which parser evidence.py should use. UNKNOWN still runs; it just parses less."""

    PYTEST = "pytest"
    JEST = "jest"
    VITEST = "vitest"
    NODE_TEST = "node-test"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Detection:
    """What to run and why, or why not. found is False exactly when the answer is NO_TESTS."""

    reason: str
    language: Language | None = None
    framework: Framework | None = None
    image: str | None = None
    command: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    fetcher: Fetcher | None = None
    requirements: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    no_tests_exit_codes: frozenset[int] = frozenset()
    notes: tuple[str, ...] = field(default=())

    @property
    def found(self) -> bool:
        return self.language is not None


@dataclass(slots=True)
class _Tree:
    python_tests: int = 0
    js_tests: int = 0
    truncated: bool = False


def _scan(root: Path) -> _Tree:
    tree = _Tree()
    seen = 0
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in _SKIP_DIRS)
        in_tests_dir = "__tests__" in Path(directory).parts
        for name in files:
            seen += 1
            if seen > MAX_FILES_SCANNED:
                tree.truncated = True
                return tree
            if _PY_TEST_FILE.match(name):
                tree.python_tests += 1
            elif _JS_TEST_FILE.match(name) or (in_tests_dir and name.endswith((".js", ".ts"))):
                tree.js_tests += 1
    return tree


def _read(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def detect(root: Path) -> Detection:
    """The one entry point. root is an extracted checkout; nothing in it is executed."""
    tree = _scan(root)
    notes: list[str] = []
    if tree.truncated:
        notes.append(f"stopped counting test files after {MAX_FILES_SCANNED} files")

    python = _detect_python(root, tree)
    javascript = _detect_javascript(root, tree)

    if python is not None and javascript is not None and javascript.found:
        # Both are real suites. The one with more test files is the repository's main one;
        # a tie goes to Python by fixed rule, since Python is the mined corpus.
        if tree.js_tests > tree.python_tests:
            chosen = javascript
            rule = f"{tree.js_tests} JS/TS test files against {tree.python_tests} Python"
        else:
            chosen = python
            rule = f"{tree.python_tests} Python test files against {tree.js_tests} JS/TS"
        notes.append(f"both a pytest suite and a package.json test script; chose by {rule}")
        return _with_notes(chosen, notes)
    if python is not None:
        return _with_notes(python, notes)
    if javascript is not None:
        return _with_notes(javascript, notes)

    for marker, language in _OTHER_LANGUAGES:
        if (root / marker).exists():
            return Detection(
                reason=f"found {marker}: {language} is outside the languages this sandbox runs",
                notes=tuple(notes),
            )
    return Detection(
        reason="no pytest configuration or Python test files, and no package.json",
        notes=tuple(notes),
    )


def _with_notes(detection: Detection, notes: list[str]) -> Detection:
    if not notes:
        return detection
    return Detection(
        reason=detection.reason,
        language=detection.language,
        framework=detection.framework,
        image=detection.image,
        command=detection.command,
        setup=detection.setup,
        fetcher=detection.fetcher,
        requirements=detection.requirements,
        env=detection.env,
        no_tests_exit_codes=detection.no_tests_exit_codes,
        notes=(*notes, *detection.notes),
    )


# Python


def _pytest_config(root: Path, pyproject: dict[str, Any]) -> str | None:
    tool = pyproject.get("tool", {})
    if isinstance(tool, dict) and "pytest" in tool:
        return "[tool.pytest] in pyproject.toml"
    if (root / "pytest.ini").is_file():
        return "pytest.ini"
    for name, section in (("setup.cfg", "tool:pytest"), ("tox.ini", "pytest")):
        text = _read(root / name)
        if text and re.search(rf"^\[{re.escape(section)}\]", text, re.MULTILINE):
            return f"[{section}] in {name}"
    return None


def _load_pyproject(root: Path, notes: list[str]) -> dict[str, Any]:
    text = _read(root / "pyproject.toml")
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        notes.append(f"pyproject.toml does not parse, ignored: {exc}")
        return {}


def _detect_python(root: Path, tree: _Tree) -> Detection | None:
    notes: list[str] = []
    pyproject = _load_pyproject(root, notes)
    config = _pytest_config(root, pyproject)
    if config is None and tree.python_tests == 0:
        return None

    if config and tree.python_tests:
        reason = f"pytest: {config}, {tree.python_tests} test files"
    elif config:
        reason = f"pytest: {config}, though no test_*.py or *_test.py file was found"
    else:
        reason = f"pytest: no configuration, {tree.python_tests} test_*.py or *_test.py files"

    installable = bool(pyproject.get("build-system") or pyproject.get("project")) or any(
        (root / name).is_file() for name in ("setup.py", "setup.cfg")
    )
    if not installable:
        notes.append("no pyproject.toml, setup.py or setup.cfg; tests run from the tree")

    requirements = _python_requirements(root, pyproject, installable, notes)
    return Detection(
        reason=reason,
        language=Language.PYTHON,
        framework=Framework.PYTEST,
        image=PYTHON_IMAGE,
        command=PYTEST_COMMAND,
        setup=PYTHON_SETUP if installable else (),
        fetcher=Fetcher.PIP,
        requirements=requirements,
        env=(("PYTHONPATH", PROJECT_SITE), *VERSION_OVERRIDES),
        no_tests_exit_codes=frozenset({PYTEST_NO_TESTS_EXIT}),
        notes=tuple(notes),
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _keyed(table: object, notes: list[str], where: str) -> list[str]:
    """The test-ish keys of a table that has them, by the order in TEST_KEYS."""
    if not isinstance(table, dict):
        return []
    keys = [key for key in TEST_KEYS if key in table]
    if not keys:
        keys = [key for key in FALLBACK_KEYS if key in table]
        if keys:
            notes.append(f"no test group in {where}, used {', '.join(keys)}")
    return keys


def _dependency_group(groups: dict[str, Any], name: str, depth: int = 0) -> list[str]:
    """PEP 735, including {include-group = ...} entries, bounded against cycles."""
    if depth > 5:
        return []
    out: list[str] = []
    for entry in groups.get(name, []):
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("include-group"), str):
            out += _dependency_group(groups, entry["include-group"], depth + 1)
    return out


def _requirements_file(root: Path, path: Path, depth: int, notes: list[str]) -> list[str]:
    text = _read(path)
    if text is None or depth > 3:
        return []
    out: list[str] = []
    for raw in text.replace("\\\n", " ").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        include = re.match(r"^(?:-r|--requirement)[ =]\s*(\S+)$", line)
        if include:
            target = (path.parent / include[1]).resolve()
            if target.is_relative_to(root.resolve()):
                out += _requirements_file(root, target, depth + 1, notes)
            continue
        if line.startswith("-"):
            # -e ., -c constraints, --index-url and friends. The project itself is built in
            # setup, and an index is not ours to change.
            continue
        # A trailing --hash or other per-requirement option is pip's syntax, not PEP 508.
        out.append(line.split(" --", 1)[0].strip())
    return out


def _poetry(pyproject: dict[str, Any], notes: list[str]) -> list[str]:
    poetry = pyproject.get("tool", {}).get("poetry", {})
    if not isinstance(poetry, dict):
        return []
    tables: list[object] = [poetry.get("dependencies")]
    groups = poetry.get("group", {})
    for key in _keyed(groups, notes, "[tool.poetry.group]"):
        tables.append(groups[key].get("dependencies") if isinstance(groups[key], dict) else None)
    tables.append(poetry.get("dev-dependencies"))
    names: list[str] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        for name, spec in table.items():
            optional = isinstance(spec, dict) and spec.get("optional") is True
            if name.lower() != "python" and not optional:
                names.append(name)
    if names:
        notes.append("Poetry constraints are not PEP 508; Poetry dependencies fetched by name")
    return names


def _python_requirements(
    root: Path, pyproject: dict[str, Any], installable: bool, notes: list[str]
) -> tuple[str, ...]:
    raw: list[str] = []
    build = pyproject.get("build-system")
    if isinstance(build, dict):
        raw += _string_list(build.get("requires"))
    elif installable:
        # PEP 517's fallback for a project that names no backend.
        raw += ["setuptools>=40.8.0", "wheel"]

    project = pyproject.get("project")
    extras: dict[str, Any] = {}
    own_name = None
    if isinstance(project, dict):
        raw += _string_list(project.get("dependencies"))
        own_name = project.get("name") if isinstance(project.get("name"), str) else None
        extras = project.get("optional-dependencies") or {}
        for key in _keyed(extras, notes, "[project.optional-dependencies]"):
            raw += _string_list(extras[key])

    groups = pyproject.get("dependency-groups")
    if isinstance(groups, dict):
        for key in _keyed(groups, notes, "[dependency-groups]"):
            raw += _dependency_group(groups, key)

    raw += _setup_cfg(root, notes)
    raw += _poetry(pyproject, notes)
    for name in REQUIREMENT_FILES:
        raw += _requirements_file(root, root / name, 0, notes)

    return _clean(raw, own_name, extras, notes)


def _setup_cfg(root: Path, notes: list[str]) -> list[str]:
    text = _read(root / "setup.cfg")
    if text is None:
        return []
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        notes.append(f"setup.cfg does not parse, ignored: {exc}")
        return []
    out: list[str] = []
    if parser.has_option("options", "install_requires"):
        out += parser.get("options", "install_requires").splitlines()
    if parser.has_section("options.extras_require"):
        section = dict(parser.items("options.extras_require"))
        for key in _keyed(section, notes, "setup.cfg extras_require"):
            out += section[key].splitlines()
    return [line.strip() for line in out if line.strip()]


def _expand_self(
    requirements: Iterable[str], own_name: str | None, extras: dict[str, Any]
) -> Iterator[str]:
    """project[test] inside the project's own extras means those extras, not a download
    of the published project from PyPI over the top of the one under review."""
    own = canonicalize_name(own_name) if own_name else None
    pending = list(requirements)
    expanded: set[str] = set()
    while pending:
        line = pending.pop(0)
        name = re.split(r"[\s\[<>=!~;@]", line, maxsplit=1)[0]
        if own is None or canonicalize_name(name) != own:
            yield line
            continue
        for extra in re.findall(r"\[([^\]]*)\]", line)[:1]:
            for key in (part.strip() for part in extra.split(",")):
                if key and key not in expanded:
                    expanded.add(key)
                    pending += _string_list(extras.get(key))


def _clean(
    raw: list[str], own_name: str | None, extras: dict[str, Any], notes: list[str]
) -> tuple[str, ...]:
    kept: list[str] = []
    refused: list[str] = []
    seen: set[str] = set()
    for line in _expand_self(raw, own_name, extras):
        if line in seen:
            continue
        seen.add(line)
        try:
            check_requirement(line)
        except SpecError:
            refused.append(line)
            continue
        kept.append(line)
    if refused:
        shown = ", ".join(refused[:3]) + (" and more" if len(refused) > 3 else "")
        notes.append(f"{len(refused)} requirements are not index packages, skipped: {shown}")
    if len(kept) > MAX_REQUIREMENTS:
        notes.append(f"{len(kept)} requirements, fetched the first {MAX_REQUIREMENTS}")
        kept = kept[:MAX_REQUIREMENTS]
    return tuple(kept)


# JavaScript and TypeScript

_LOCKFILES = (
    ("package-lock.json", "npm"),
    ("npm-shrinkwrap.json", "npm"),
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("bun.lockb", "bun"),
    ("bun.lock", "bun"),
)

_MANAGER_ORDER = ("npm", "pnpm", "yarn", "bun")


def _detect_javascript(root: Path, tree: _Tree) -> Detection | None:
    text = _read(root / "package.json")
    if text is None:
        return None
    try:
        package: Any = json.loads(text)
    except json.JSONDecodeError:
        return Detection(reason="package.json is not valid JSON")
    if not isinstance(package, dict):
        return Detection(reason="package.json is not a JSON object")

    scripts = package.get("scripts")
    script = scripts.get("test") if isinstance(scripts, dict) else None
    if not isinstance(script, str) or not script.strip():
        return Detection(reason="package.json has no scripts.test")
    if _NPM_PLACEHOLDER.search(script):
        return Detection(reason="scripts.test is npm's placeholder, not a suite")

    notes: list[str] = []
    present = [manager for name, manager in _LOCKFILES if (root / name).is_file()]
    declared = _declared_manager(package)
    manager, why = _choose_manager(present, declared)
    notes.append(why)

    if manager == "bun":
        return Detection(reason="Bun is not a package manager this sandbox runs", notes=(why,))
    if manager == "yarn" and _is_yarn_berry(root, declared):
        return Detection(
            reason="Yarn 2 and later run a yarn release committed to the repository, which "
            "is repository code executing with network during the install; not run",
            notes=(why,),
        )

    if manager == "npm":
        fetcher = Fetcher.NPM_CI if "npm" in present else Fetcher.NPM_INSTALL
    else:
        fetcher = Fetcher.PNPM if manager == "pnpm" else Fetcher.YARN

    framework = _js_framework(script, package)
    return Detection(
        reason=f"package.json scripts.test run with {manager}: {script[:120]}",
        language=Language.JAVASCRIPT,
        framework=framework,
        image=NODE_IMAGE,
        command=(manager, "test"),
        fetcher=fetcher,
        notes=tuple(notes),
    )


def _declared_manager(package: dict[str, Any]) -> tuple[str, str] | None:
    """packageManager, as corepack reads it: name@version."""
    value = package.get("packageManager")
    if not isinstance(value, str):
        return None
    match = re.match(r"^(npm|pnpm|yarn|bun)@(\S+)", value)
    return (match[1], match[2]) if match else None


def _choose_manager(present: list[str], declared: tuple[str, str] | None) -> tuple[str, str]:
    kinds = sorted(set(present), key=_MANAGER_ORDER.index)
    if declared is not None:
        name = declared[0]
        others = [kind for kind in kinds if kind != name]
        extra = f"; ignored {', '.join(others)} lockfiles" if others else ""
        return name, f"packageManager names {name}{extra}"
    if not kinds:
        return "npm", "no lockfile, so npm install resolves afresh and may not match the author's"
    if len(kinds) == 1:
        return kinds[0], f"{kinds[0]} lockfile"
    return kinds[0], (
        f"lockfiles for {', '.join(kinds)} and no packageManager field; "
        f"chose {kinds[0]} by the fixed order {', '.join(_MANAGER_ORDER)}"
    )


def _is_yarn_berry(root: Path, declared: tuple[str, str] | None) -> bool:
    if declared is not None and declared[0] == "yarn":
        return not declared[1].startswith("1.")
    if (root / ".yarnrc.yml").exists():
        return True
    lock = _read(root / "yarn.lock") or ""
    return "__metadata:" in lock[:2000]


def _js_framework(script: str, package: dict[str, Any]) -> Framework:
    if "vitest" in script:
        return Framework.VITEST
    if "jest" in script or "react-scripts test" in script:
        return Framework.JEST
    if re.search(r"\bnode\b.*--test\b", script):
        return Framework.NODE_TEST
    deps: set[str] = set()
    for key in ("devDependencies", "dependencies"):
        table = package.get(key)
        if isinstance(table, dict):
            deps.update(table)
    if "vitest" in deps:
        return Framework.VITEST
    if "jest" in deps:
        return Framework.JEST
    return Framework.UNKNOWN
