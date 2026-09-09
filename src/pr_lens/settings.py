from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """What the receiver needs, and nothing more.

    There is no DATABASE_URL here on purpose. The receiver never opens a database
    connection, so it must not be able to. Jobs that do need Postgres read the environment
    themselves. Field names are the exact secret names; renaming one renames it in Vercel
    and in the checklist too.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    gh_webhook_secret: str
    gh_dispatch_token: str

    # Config, not a constant, so a fork or the testbed can point somewhere else.
    dispatch_repo: str = "SURYAPRASATHJP/pr-lens"
    dispatch_event_type: str = "pr_event"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
