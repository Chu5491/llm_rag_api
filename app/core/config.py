# app/core/config.py
from pydantic_settings import BaseSettings
import sys


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # 정의되지 않은 환경변수는 무시

    # =========================================================
    # 필수 시스템 환경변수 (DB 접속 정보 등 - .env 필수)
    # =========================================================
    MCP_DB_CON: str
    VECTOR_DB_CON: str

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 10

    MCP_SERVER_COMMAND: str = sys.executable
    MCP_SERVER: str = "mcp_server.py"

    # [Path Settings]
    UPLOADS_DIR: str = "uploads"
    UPLOADS_VECTOR_INDEX_PATH: str = "data/index.faiss"
    UPLOADS_VECTOR_META_PATH: str = "data/index_meta.json"
    FIGMA_VECTOR_INDEX_PATH: str = "data/figma.faiss"
    FIGMA_VECTOR_META_PATH: str = "data/figma_meta.json"

    # [RAG Settings]
    RAG_DEFAULT_QUERY: str = "시스템의 주요 기능 및 예외 처리"
    RAG_TOP_K: int = 5

    # [Figma Settings]
    FIGMA_CHUNK_MODE: str = "section_only"


# 전역 설정 인스턴스 (초기에는 .env 값만 가짐)
_settings_instance = Settings()


def get_settings() -> Settings:
    """설정 인스턴스를 반환 (싱글톤)"""
    return _settings_instance
