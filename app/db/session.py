"""Database connections.

The API uses the async engine (asyncpg). Batch pipelines use a plain synchronous
engine (psycopg), which is simpler and just as fast for bulk loads.
"""
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import settings


def sync_url(url: str) -> str:
    """Turn the async database URL into its synchronous twin."""
    return url.replace("+asyncpg", "+psycopg")


def make_sync_engine() -> Engine:
    return create_engine(sync_url(settings.database_url), pool_pre_ping=True)


def make_async_engine() -> AsyncEngine:
    return create_async_engine(settings.database_url, pool_pre_ping=True)


async_engine = make_async_engine()
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)