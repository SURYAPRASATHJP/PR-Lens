from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every value comes from the environment. Nothing is read from a file.

    Field names map to the exact secret names agreed in notes/handoff-to-claude-code.md.
    Renaming a field renames a secret in three places: Actions, the Space, and the
    checklist, so don't.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    gh_webhook_secret: str
    gh_dispatch_token: str
    database_url: str

    # The repo that receives repository_dispatch. Config, not a constant, because
    # a fork or a test target changes it without touching code.
    dispatch_repo: str = "SURYAPRASATHJP/pr-lens"
    dispatch_event_type: str = "pr_event"

    # Minted per installation to post comments. Phase 4 needs these; Phase 0 does not,
    # so the Space can boot before the App exists.
    gh_app_id: str | None = None
    gh_app_private_key: str | None = None

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
