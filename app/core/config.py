from pydantic_settings import BaseSettings
import sys


# 애플리케이션 전역 설정 클래스
class Settings(BaseSettings):
    # 환경변수 로딩 파일 지정(.env)
    class Config:
        # dotenv 사용 허용
        env_file = ".env"
        # 케이스 민감도 설정
        case_sensitive = True

    # Ollama 서버 베이스 URL
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Ollama HTTP 타임아웃(초)
    OLLAMA_TIMEOUT: int = 10

    # 임베딩 모델 이름 (예: 'nomic-embed-text', 'bona/bge-m3-korean' 등)
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # LLM 모델 (chat/generate용) ← 새로 추가
    LLM_MODEL: str = "llama3.1:8b"

    # rag 대상 파일 경로
    UPLOADS_DIR: str = "uploads"

    # FAISS 벡터 파일 경로
    UPLOADS_VECTOR_INDEX_PATH: str = "data/index.faiss"

    FIGMA_VECTOR_INDEX_PATH: str = "data/figma.faiss"

    # 메타데이터(JSON) 파일 경로
    UPLOADS_VECTOR_META_PATH: str = "data/index_meta.json"

    FIGMA_VECTOR_META_PATH: str = "data/figma_meta.json"

    # 테스트케이스 생성용 기본 쿼리 (RAG 검색 용도)
    RAG_DEFAULT_QUERY: str = (
        "이 시스템의 요구사항과 화면 정의, 예외 상황을 잘 포함하는 부분"
    )

    # 테스트케이스 개수 (rules 안의 n 용도)
    RAG_TC_COUNT: int = 10

    # rag 상위 k개 문서 사용
    RAG_TOP_K: int = 5

    # rag 배치 사이즈
    RAG_BATCH_SIZE: int = 3

    # 테스트케이스 ID prefix (TC-001, TC-002 ...)
    RAG_TC_ID_PREFIX: str = "TC"

    # MCP DB 연결 정보
    MCP_DB_CON: str = "postgresql://tbell:tbell@tbell:5432/tbell"

    # Vector DB 연결 정보
    VECTOR_DB_CON: str = "postgresql://tbell:tbell@tbell:5432/tbell"

    # MCP 서버 설정
    MCP_SERVER_COMMAND: str = sys.executable  # 현재 파이썬

    MCP_SERVER: str = "mcp_server.py"

    # Figma 설정
    FIGMA_ENABLED: bool = "true"
    FIGMA_API_TOKEN: str = ""
    FIGMA_URL: str = ""

    # Figma 청크 모드: "section_only", "screen_only", "both"
    # - section_only: 섹션 단위 청크만 (추천, 검색 정확도 높음)
    # - screen_only: 화면 전체 청크만 (청크 수 최소)
    # - both: 둘 다 생성 (중복 있음)
    FIGMA_CHUNK_MODE: str = "section_only"


# 싱글톤 성격의 설정 인스턴스 함수
def get_settings() -> Settings:
    # Settings 인스턴스 생성 및 반환
    return Settings()
