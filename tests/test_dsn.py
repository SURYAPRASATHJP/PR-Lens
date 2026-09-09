from pr_lens.db.pool import normalise_dsn


def test_drops_parameters_asyncpg_rejects() -> None:
    neon = (
        "postgresql://user:pw@ep-cool-name-pooler.us-east-2.aws.neon.tech/pr_lens"
        "?sslmode=require&channel_binding=require"
    )
    assert normalise_dsn(neon) == (
        "postgresql://user:pw@ep-cool-name-pooler.us-east-2.aws.neon.tech/pr_lens?sslmode=require"
    )


def test_keeps_sslmode_which_asyncpg_understands() -> None:
    assert "sslmode=require" in normalise_dsn("postgresql://h/db?sslmode=require")


def test_leaves_a_plain_local_dsn_alone() -> None:
    plain = "postgresql://postgres:postgres@localhost:5432/postgres"
    assert normalise_dsn(plain) == plain
