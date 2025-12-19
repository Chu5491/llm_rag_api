# app/db/database.py
from typing import Generator
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import get_settings

settings = get_settings()


def create_database_connection(db_url: str):
    """데이터베이스 URL을 받아 엔진과 세션을 생성합니다."""
    # pg8000 사용을 위해 URL 변환
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+pg8000://", 1)

    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine, SessionLocal, Base


# ===== Vector DB (기본 DB) =====
vector_engine, VectorSessionLocal, VectorBase = create_database_connection(
    settings.VECTOR_DB_CON
)


def get_vector_db() -> Generator[Session, None, None]:
    """
    Vector DB 세션을 가져옵니다.
    FastAPI의 Depends와 함께 사용할 수 있도록 제네레이터로 구현되었습니다.
    """
    db = VectorSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===== MCP DB =====
mcp_engine, MCPSessionLocal, MCPBase = create_database_connection(settings.MCP_DB_CON)


# MCP DB Connection (pg8000)
def get_mcp_db_connection():
    """MCP DB에 대한 pg8000 연결을 반환합니다."""
    import pg8000

    # Parse the connection string
    r = urlparse(settings.MCP_DB_CON)
    return pg8000.connect(
        user=r.username,
        password=r.password,
        host=r.hostname or "localhost",
        port=r.port or 5432,
        database=(r.path or "").lstrip("/") or None,
    )


# 이전 버전과의 호환성을 위해 get_db 유지 (기본값으로 Vector DB 사용)
get_vector_db = get_vector_db
get_mcp_db = get_mcp_db_connection
