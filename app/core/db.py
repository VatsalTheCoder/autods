"""Database engine, session factory and declarative base.

No tables are defined yet -- Section 1 adds ``users``, ``jobs`` and
``artifacts`` through an Alembic migration. This module just establishes the
connection machinery so the health check has something real to test.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

engine = create_engine(
    str(settings.database_url),
    # Verifies a pooled connection is still alive before handing it out.
    # Without this, connections idle overnight fail the next morning.
    pool_pre_ping=True,
    echo=settings.debug and settings.is_local,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a session that is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def database_healthy() -> bool:
    """Cheap connectivity check, used by the /health endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False
