import json
from pathlib import Path

import pytest

from pr_lens.sandbox import images
from pr_lens.sandbox.detect import (
    PYTEST_COMMAND,
    PYTHON_SETUP,
    Detection,
    Framework,
    Language,
    detect,
)
from pr_lens.sandbox.spec import Fetcher, FetchSpec, SandboxSpec


def write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def package(test: str | None = "jest", **extra: object) -> str:
    body: dict[str, object] = {"name": "demo", "version": "1.0.0", **extra}
    if test is not None:
        body["scripts"] = {"test": test}
    return json.dumps(body)


PYPROJECT = """
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "demo"
dynamic = ["version"]
dependencies = ["httpx>=0.28", "attrs"]

[project.optional-dependencies]
fast = ["orjson"]
test = ["demo[fast]", "pytest-asyncio", "respx"]
docs = ["sphinx"]

[dependency-groups]
lint = ["ruff"]
test = ["coverage", {include-group = "lint"}]

[tool.pytest.ini_options]
addopts = "-ra"
"""


def test_an_empty_repository_has_no_tests_and_says_so(tmp_path: Path) -> None:
    detection = detect(tmp_path)
    assert not detection.found
    assert "no pytest configuration" in detection.reason
    assert detection.command == ()


def test_another_language_is_named_in_the_reason(tmp_path: Path) -> None:
    write(tmp_path, {"Cargo.toml": "[package]\nname = 'x'\n", "src/lib.rs": ""})
    detection = detect(tmp_path)
    assert not detection.found
    assert "Rust" in detection.reason


def test_a_full_pyproject_yields_pytest_with_its_test_dependencies(tmp_path: Path) -> None:
    write(tmp_path, {"pyproject.toml": PYPROJECT, "tests/test_a.py": "", "tests/test_b.py": ""})
    detection = detect(tmp_path)

    assert detection.language is Language.PYTHON
    assert detection.framework is Framework.PYTEST
    assert detection.image == images.PYTHON_IMAGE
    assert detection.command == PYTEST_COMMAND
    assert detection.setup == PYTHON_SETUP
    assert detection.fetcher is Fetcher.PIP
    assert detection.no_tests_exit_codes == frozenset({5})
    assert "[tool.pytest] in pyproject.toml" in detection.reason
    assert "2 test files" in detection.reason
    # Build backend first, then runtime, then every extra in file order with demo[fast]
    # standing for its own extra rather than a download of the published demo, then the
    # PEP 735 test group and the group it includes.
    assert detection.requirements == (
        "hatchling",
        "hatch-vcs",
        "httpx>=0.28",
        "attrs",
        "orjson",
        "pytest-asyncio",
        "respx",
        "sphinx",
        "coverage",
        "ruff",
    )
    assert "read every extra: fast, test, docs" in detection.notes
    assert "read dependency groups: test" in detection.notes
    assert ("SETUPTOOLS_SCM_PRETEND_VERSION", "9999.0.0") in detection.env


def test_tests_without_any_packaging_run_from_the_tree(tmp_path: Path) -> None:
    write(tmp_path, {"test_thing.py": "def test_x(): pass\n"})
    detection = detect(tmp_path)
    assert detection.language is Language.PYTHON
    assert detection.setup == ()
    assert detection.requirements == ()
    assert "no configuration, 1 test_*.py" in detection.reason
    assert any("tests run from the tree" in note for note in detection.notes)


def test_pytest_configured_in_tox_ini_counts(tmp_path: Path) -> None:
    write(tmp_path, {"tox.ini": "[tox]\nenvlist = py312\n\n[pytest]\ntestpaths = t\n"})
    detection = detect(tmp_path)
    assert detection.found
    assert "[pytest] in tox.ini" in detection.reason


def test_requirements_files_are_read_and_hazards_skipped_with_a_note(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "setup.py": "from setuptools import setup\nsetup()\n",
            "requirements.txt": "requests==2.32.0 --hash=sha256:abc\n-r requirements/base.txt\n",
            "requirements/base.txt": (
                "# comment\nclick>=8  # inline\n-e .\n--index-url https://evil.example/simple\n"
                "evil @ https://evil.example/evil.whl\n-r ../../outside.txt\n"
            ),
            "requirements-dev.txt": "pytest-mock\n",
            "tests/test_x.py": "",
        },
    )
    detection = detect(tmp_path)
    assert detection.requirements == (
        "setuptools>=40.8.0",
        "wheel",
        "requests==2.32.0",
        "click>=8",
        "pytest-mock",
    )
    assert any("1 requirements are not index packages" in note for note in detection.notes)


def test_dev_and_uv_default_groups_are_read_beside_the_test_group(tmp_path: Path) -> None:
    """urllib3's shape, measured 14 Sep 2026: its conftest imports trustme, which only the
    dev group names. Reading the test group alone ran none of its suite."""
    write(
        tmp_path,
        {
            "pyproject.toml": (
                "[project]\nname = 'x'\nversion = '1'\n"
                "[dependency-groups]\n"
                "test = ['pytest-timeout']\n"
                "dev = [{include-group = 'test'}, 'trustme']\n"
                "integrations = ['fastapi']\n"
                "docs = ['sphinx']\n"
                "[tool.uv]\ndefault-groups = ['dev', 'integrations']\n"
            ),
            "tests/test_x.py": "",
        },
    )
    detection = detect(tmp_path)
    # A [project] with no build backend gets PEP 517's setuptools fallback first.
    assert detection.requirements == (
        "setuptools>=40.8.0",
        "wheel",
        "pytest-timeout",
        "trustme",
        "fastapi",
    )
    assert "read dependency groups: test, dev, integrations" in detection.notes


def test_a_dev_requirements_file_with_an_underscore_is_read(tmp_path: Path) -> None:
    """redis-py's shape: no groups at all, and dev_requirements.txt."""
    write(tmp_path, {"dev_requirements.txt": "pytest-asyncio\n", "tests/test_x.py": ""})
    assert detect(tmp_path).requirements == ("pytest-asyncio",)


def test_pytest_keeps_going_past_a_module_that_will_not_import() -> None:
    assert "--continue-on-collection-errors" in PYTEST_COMMAND


def test_setup_cfg_and_poetry_contribute_their_static_dependencies(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "setup.cfg": (
                "[options]\ninstall_requires =\n    six\n    packaging>=20\n"
                "[options.extras_require]\ntesting =\n    hypothesis\n"
            ),
            "pyproject.toml": (
                "[tool.poetry.dependencies]\npython = '^3.12'\nrich = '^13'\n"
                "maybe = {version = '1', optional = true}\n"
                "[tool.poetry.group.test.dependencies]\nfreezegun = '*'\n"
            ),
            "tests/test_x.py": "",
        },
    )
    detection = detect(tmp_path)
    for name in ("six", "packaging>=20", "hypothesis", "rich", "freezegun"):
        assert name in detection.requirements
    assert "maybe" not in detection.requirements
    assert "python" not in detection.requirements


def test_a_broken_pyproject_is_noted_and_does_not_hide_the_tests(tmp_path: Path) -> None:
    write(tmp_path, {"pyproject.toml": "[project\nname=", "tests/test_x.py": ""})
    detection = detect(tmp_path)
    assert detection.found
    assert any("does not parse" in note for note in detection.notes)


def test_a_package_json_without_a_test_script_has_no_tests(tmp_path: Path) -> None:
    write(tmp_path, {"package.json": package(test=None)})
    assert detect(tmp_path).reason == "package.json has no scripts.test"


def test_npms_placeholder_script_is_not_a_suite(tmp_path: Path) -> None:
    placeholder = 'echo "Error: no test specified" && exit 1'
    write(tmp_path, {"package.json": package(test=placeholder)})
    detection = detect(tmp_path)
    assert not detection.found
    assert "placeholder" in detection.reason


def test_an_npm_lockfile_means_npm_ci(tmp_path: Path) -> None:
    write(tmp_path, {"package.json": package("jest --ci"), "package-lock.json": "{}"})
    detection = detect(tmp_path)
    assert detection.language is Language.JAVASCRIPT
    assert detection.framework is Framework.JEST
    assert detection.fetcher is Fetcher.NPM_CI
    assert detection.command == ("npm", "test")
    assert detection.image == images.NODE_IMAGE


def test_no_lockfile_means_npm_install_and_a_note_that_it_may_differ(tmp_path: Path) -> None:
    write(tmp_path, {"package.json": package("vitest run")})
    detection = detect(tmp_path)
    assert detection.fetcher is Fetcher.NPM_INSTALL
    assert detection.framework is Framework.VITEST
    assert any("no lockfile" in note for note in detection.notes)


def test_pnpm_and_yarn_classic_lockfiles_pick_their_manager(tmp_path: Path) -> None:
    pnpm = write(tmp_path / "p", {"package.json": package(), "pnpm-lock.yaml": ""})
    yarn = write(tmp_path / "y", {"package.json": package(), "yarn.lock": "# yarn v1\n"})
    assert (detect(pnpm).fetcher, detect(pnpm).command) == (Fetcher.PNPM, ("pnpm", "test"))
    assert (detect(yarn).fetcher, detect(yarn).command) == (Fetcher.YARN, ("yarn", "test"))


def test_two_lockfiles_resolve_by_a_fixed_order_and_record_it(tmp_path: Path) -> None:
    write(tmp_path, {"package.json": package(), "yarn.lock": "", "package-lock.json": "{}"})
    detection = detect(tmp_path)
    assert detection.command == ("npm", "test")
    assert any("fixed order" in note for note in detection.notes)


def test_package_manager_field_outranks_the_lockfiles(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "package.json": package(packageManager="pnpm@9.1.0"),
            "package-lock.json": "{}",
            "pnpm-lock.yaml": "",
        },
    )
    detection = detect(tmp_path)
    assert detection.command == ("pnpm", "test")
    assert any("ignored npm" in note for note in detection.notes)


@pytest.mark.parametrize(
    "files",
    [
        {"package.json": package(packageManager="yarn@4.5.0"), "yarn.lock": ""},
        {"package.json": package(), "yarn.lock": "", ".yarnrc.yml": "yarnPath: x.cjs\n"},
        {"package.json": package(), "yarn.lock": "__metadata:\n  version: 8\n"},
    ],
    ids=["package-manager", "yarnrc", "lockfile-metadata"],
)
def test_yarn_berry_is_refused_because_it_runs_a_committed_binary(
    tmp_path: Path, files: dict[str, str]
) -> None:
    write(tmp_path, files)
    detection = detect(tmp_path)
    assert not detection.found
    assert "Yarn 2" in detection.reason


def test_bun_is_refused(tmp_path: Path) -> None:
    write(tmp_path, {"package.json": package(), "bun.lockb": ""})
    assert "Bun" in detect(tmp_path).reason


def test_node_test_and_dependency_based_frameworks(tmp_path: Path) -> None:
    node = write(tmp_path / "n", {"package.json": package("node --test test/")})
    dep = write(
        tmp_path / "d",
        {"package.json": package("npm run unit", devDependencies={"vitest": "^2"})},
    )
    other = write(tmp_path / "o", {"package.json": package("mocha")})
    assert detect(node).framework is Framework.NODE_TEST
    assert detect(dep).framework is Framework.VITEST
    assert detect(other).framework is Framework.UNKNOWN


def test_invalid_package_json_has_no_tests(tmp_path: Path) -> None:
    write(tmp_path, {"package.json": "{not json"})
    assert detect(tmp_path).reason == "package.json is not valid JSON"


def test_both_languages_pick_the_bigger_suite_and_say_so(tmp_path: Path) -> None:
    js_heavy = write(
        tmp_path / "js",
        {
            "package.json": package(),
            "package-lock.json": "{}",
            "tests/test_a.py": "",
            "src/a.test.ts": "",
            "src/b.test.ts": "",
        },
    )
    tie = write(
        tmp_path / "tie",
        {"package.json": package(), "tests/test_a.py": "", "src/a.test.js": ""},
    )
    assert detect(js_heavy).language is Language.JAVASCRIPT
    assert detect(tie).language is Language.PYTHON
    assert any("both a pytest suite" in note for note in detect(tie).notes)


def test_a_javascript_placeholder_does_not_shadow_a_python_suite(tmp_path: Path) -> None:
    write(
        tmp_path,
        {"package.json": package(test="echo no test specified"), "tests/test_a.py": ""},
    )
    assert detect(tmp_path).language is Language.PYTHON


def test_node_modules_and_virtualenvs_are_not_counted(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "node_modules/x/test_y.py": "",
            ".venv/lib/test_z.py": "",
            "node_modules/x/a.test.js": "",
        },
    )
    assert not detect(tmp_path).found


def test_the_same_tree_always_gives_the_same_answer(tmp_path: Path) -> None:
    write(tmp_path, {"pyproject.toml": PYPROJECT, "tests/test_a.py": "", "package.json": "{}"})
    first: Detection = detect(tmp_path)
    assert all(detect(tmp_path) == first for _ in range(5))


def test_every_detection_builds_a_spec_the_sandbox_accepts(tmp_path: Path) -> None:
    """A detection that produced an invalid spec would fail at run time, per repository."""
    write(tmp_path, {"pyproject.toml": PYPROJECT, "tests/test_a.py": ""})
    detection = detect(tmp_path)
    assert detection.image is not None and detection.fetcher is not None
    FetchSpec(detection.image, detection.fetcher, tmp_path, "pr-lens-x", detection.requirements)
    SandboxSpec(
        image=detection.image,
        command=detection.command,
        source_dir=tmp_path,
        setup=detection.setup,
        env=detection.env,
    )


def test_the_pinned_images_are_real_digests() -> None:
    for image in (images.PYTHON_IMAGE, images.NODE_IMAGE):
        assert image.startswith("ghcr.io/suryaprasathjp/")
        digest = image.split("@sha256:")[1]
        assert len(digest) == 64
        assert set(digest) != {"0"}


# The Phase 3 dependency review's asymmetry: pip requirements naming a URL were refused,
# while a JS lockfile naming one was honoured by a fetch step that has network.

LEFT_PAD = "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz"
NPM_LOCK = {
    "lockfileVersion": 3,
    "packages": {
        "": {"name": "demo"},
        "node_modules/left-pad": {"version": "1.3.0", "resolved": LEFT_PAD},
    },
}


@pytest.mark.parametrize(
    ("files", "named"),
    [
        (
            {
                "package-lock.json": json.dumps(
                    {
                        "lockfileVersion": 3,
                        "packages": {
                            "node_modules/evil": {
                                "resolved": "git+ssh://git@example.com/evil.git#abc",
                            }
                        },
                    }
                )
            },
            "node_modules/evil from git+ssh://git@example.com/evil.git#abc",
        ),
        (
            {
                "package-lock.json": json.dumps(
                    {
                        "lockfileVersion": 1,
                        "dependencies": {
                            "mirror": {"resolved": "https://registry.npmmirror.com/m.tgz"}
                        },
                    }
                )
            },
            "mirror from https://registry.npmmirror.com/m.tgz",
        ),
        (
            {
                "pnpm-lock.yaml": "packages:\n  /tool@1.0.0:\n    resolution: "
                "{tarball: https://codeload.github.com/o/tool/tar.gz/abc}\n"
            },
            "tarball https://codeload.github.com/o/tool/tar.gz/abc",
        ),
        (
            {
                "yarn.lock": '# yarn lockfile v1\n\ntool@^1:\n  version "1.0.0"\n'
                '  resolved "https://evil.example/tool-1.0.0.tgz#abc"\n'
            },
            "resolved https://evil.example/tool-1.0.0.tgz#abc",
        ),
    ],
    ids=["npm git", "npm v1 mirror", "pnpm tarball", "yarn host"],
)
def test_a_lockfile_resolving_outside_the_registry_is_not_fetched(
    tmp_path: Path, files: dict[str, str], named: str
) -> None:
    detection = detect(write(tmp_path, {"package.json": package(), **files}))
    assert not detection.found
    assert "outside the public registry" in detection.reason
    assert named in detection.reason


@pytest.mark.parametrize(
    "spec", ["github:o/tool", "o/tool", "git+https://example.com/t.git", "https://x.example/t.tgz"]
)
def test_package_json_naming_a_remote_source_is_not_fetched(tmp_path: Path, spec: str) -> None:
    """With no lockfile npm install resolves package.json itself, so it is read too."""
    detection = detect(write(tmp_path, {"package.json": package(dependencies={"tool": spec})}))
    assert not detection.found
    assert f"tool from {spec}" in detection.reason


def test_registry_and_local_sources_are_fetched_as_before(tmp_path: Path) -> None:
    deps = {"left-pad": "^1.3.0", "alias": "npm:@scope/real@2", "local": "file:../local"}
    root = write(
        tmp_path,
        {
            "package.json": package(dependencies=deps, devDependencies={"ws": "workspace:*"}),
            "package-lock.json": json.dumps(NPM_LOCK),
        },
    )
    detection = detect(root)
    assert detection.found
    assert detection.fetcher is Fetcher.NPM_CI


def test_a_lockfile_over_the_manifest_cap_is_still_checked(tmp_path: Path) -> None:
    """A real package-lock.json is often past the 1 MB manifest cap. Skipping it would pass
    an unchecked lockfile, which is the hole this check exists to close."""
    lock = {
        "lockfileVersion": 3,
        "packages": {
            **{
                f"node_modules/p{n}": {
                    "resolved": LEFT_PAD,
                    "pad": "x" * 200,
                }
                for n in range(6000)
            },
            "node_modules/evil": {"resolved": "https://evil.example/e.tgz"},
        },
    }
    text = json.dumps(lock)
    assert len(text) > 1_000_000
    detection = detect(write(tmp_path, {"package.json": package(), "package-lock.json": text}))
    assert not detection.found
    assert "node_modules/evil from https://evil.example/e.tgz" in detection.reason
