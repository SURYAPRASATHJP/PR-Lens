"""Guards on the one property the receiver has to keep.

It runs on a free serverless tier, in front of a database on another free tier, and the
whole point of the split is that the request path never touches Postgres. That is easy to
say in a comment and easy to undo with one import, so it is asserted here instead.
"""

import subprocess
import sys

import pytest

from pr_lens.settings import Settings


def test_importing_the_receiver_does_not_import_asyncpg() -> None:
    # A subprocess, because by the time this test file runs, the suite has already
    # imported asyncpg for the Actions-side tests.
    # pr_lens.api.main is the receiver itself; api/index.py only re-exports it.
    code = (
        "import sys; import pr_lens.api.main; "
        "sys.exit(1 if any(m.startswith('asyncpg') for m in sys.modules) else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        "the receiver pulled in asyncpg. Something on its import path reaches the "
        f"database.\n{result.stderr}"
    )


def test_settings_have_no_database_url() -> None:
    assert "database_url" not in Settings.model_fields


def test_settings_need_only_what_vercel_holds() -> None:
    required = {n for n, f in Settings.model_fields.items() if f.is_required()}
    assert required == {"gh_webhook_secret", "gh_dispatch_token"}


def test_the_receiver_boots_with_only_those_two_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("DATABASE_URL", "GH_APP_ID", "GH_APP_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GH_WEBHOOK_SECRET", "x")
    monkeypatch.setenv("GH_DISPATCH_TOKEN", "y")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.dispatch_repo == "SURYAPRASATHJP/pr-lens"
