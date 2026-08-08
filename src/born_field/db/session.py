"""Engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from born_field.config import Settings
from born_field.db.models import Base
from born_field.logging import get_logger

log = get_logger(__name__)


def build_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Create an engine against the configured database."""
    return create_engine(url or Settings().database_url, echo=echo, pool_pre_ping=True)


def init_schema(engine: Engine) -> None:
    """Enable PostGIS and create tables."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    log.info("schema_ready", tables=sorted(Base.metadata.tables))


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A transactional scope that commits on success and rolls back on error."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
