import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("database")

# Ensure local data directory exists if using SQLite
if settings.DATABASE_URL.startswith("sqlite"):
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

# Create SQLAlchemy engine
# SQLite needs check_same_thread=False for multithreading in Streamlit/FastAPI
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

from sqlalchemy import event

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
Base = declarative_base()


@contextmanager
def get_db():
    """
    Context manager for transactional database sessions.
    Automatically commits on success or rolls back on exception.
    Safely commits when Streamlit triggers script rerun exceptions.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        # Check if exception is Streamlit's internal RerunException or StopException
        type_name = type(e).__name__
        if "Rerun" in type_name or "Stop" in type_name:
            try:
                db.commit()
            except Exception:
                db.rollback()
            raise e
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise e
    finally:
        db.close()


def init_db():
    """
    Creates all tables in the database if they don't already exist,
    and runs initial seed data setup.
    """
    from app.db.models import Base
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise e
