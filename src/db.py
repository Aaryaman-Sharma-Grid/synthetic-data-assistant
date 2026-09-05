"""PostgreSQL connection helpers."""

from sqlalchemy import create_engine, text

from src.config import Settings


def check_database_connection(settings: Settings) -> bool:
    """Return whether PostgreSQL is reachable without raising in the UI."""
    try:
        with create_engine(settings.database_url, pool_pre_ping=True).connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

