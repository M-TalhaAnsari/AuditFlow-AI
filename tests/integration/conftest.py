import pytest
from testcontainers.community.postgres import PostgresContainer
from alembic.config import Config
from alembic import command

@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:16") as pg:
        yield pg

@pytest.fixture(scope="session")
def database_url(postgres_container):
    url = postgres_container.get_connection_url()
    # testcontainers gives you 'postgresql+psycopg2://...' - normalize if your
    # code expects a plain 'postgresql://' DSN
    return url.replace("postgresql+psycopg2://", "postgresql://")

@pytest.fixture(scope="session", autouse=True)
def run_migrations(database_url):
    import os
    os.environ["DATABASE_URL"] = database_url
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")