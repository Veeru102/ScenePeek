"""Integration tests run against a throwaway database on the local compose Postgres."""

import os

import psycopg
import pytest
from sqlalchemy import create_engine, text

import scenepeek.models  # noqa: F401
from scenepeek.core.db import Base

ADMIN_URL = os.environ.get("TEST_ADMIN_URL", "postgresql://scenepeek:scenepeek@localhost:5434/postgres")
TEST_DB = "scenepeek_test"
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


@pytest.fixture(scope="session")
def test_db_url():
    with psycopg.connect(ADMIN_URL, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {TEST_DB}")
    engine = create_engine("postgresql+psycopg://" + TEST_URL.split("://", 1)[1])
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        Base.metadata.create_all(conn)
    engine.dispose()
    yield "postgresql+psycopg://" + TEST_URL.split("://", 1)[1]
    with psycopg.connect(ADMIN_URL, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


@pytest.fixture
def engine(test_db_url):
    eng = create_engine(test_db_url)
    with eng.begin() as conn:
        for t in (
            "jobs",
            "embeddings",
            "topics",
            "frames",
            "segments",
            "utterances",
            "video_chunks",
            "videos",
            "experiment_results",
            "experiments",
            "dataset_queries",
            "dataset_videos",
            "datasets",
            "index_versions",
        ):
            conn.execute(text(f"TRUNCATE {t} CASCADE"))
    yield eng
    eng.dispose()
