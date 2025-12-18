# FastAPI 임포트
from fastapi import FastAPI

# Lifespan 관리를 위한 asynccontextmanager 임포트
from contextlib import asynccontextmanager

# GZip 미들웨어 임포트
from starlette.middleware.gzip import GZipMiddleware

# CORS 미들웨어 임포트
from fastapi.middleware.cors import CORSMiddleware

# 라우터 임포트
from app.api.v1.endpoints import ollama as ollama_router
from app.api.v1.endpoints import rag as rag_router
from app.api.v1.endpoints import mcp as mcp_router
from app.api.v1.endpoints import figma as figma_router
from app.api.v1.endpoints import rag_faiss as rag_faiss_router

# RAG 벡터 스토어 임포트
# FAISS 버전
# from app.services.file_rag_store import file_rag_vector_store
# from app.services.figma_rag_store import figma_rag_vector_store
# PG 버전
from app.services.file_rag_store_pg import file_rag_vector_store_pg
from app.services.figma_rag_store_pg import figma_rag_vector_store_pg

# 로깅 임포트
from app.core.logging import setup_logging, get_logger

# 가장 먼저 로깅 초기화
setup_logging()
logger = get_logger(__name__)


# 애플리케이션 수명주기(lifespan) 이벤트 핸들러 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 스타트업
    logger.info("=" * 60)
    logger.info("LLM RAG API 서버 시작")
    logger.info("=" * 60)
    # 서버 시작 시 한 번 실행할 초기화 로직
    # await file_rag_vector_store.ensure_vector_store()
    # await figma_rag_vector_store.ensure_vector_store()
    await file_rag_vector_store_pg.ensure_vector_store()
    await figma_rag_vector_store_pg.ensure_vector_store()

    yield
    # 셧다운
    logger.info("=" * 60)
    logger.info("LLM RAG API 서버 종료")
    logger.info("=" * 60)


# 애플리케이션 인스턴스 생성 (lifespan 등록)
app = FastAPI(
    title="RAG/Ollama API",
    version="0.1.0",
    lifespan=lifespan,
)

# GZip 미들웨어 추가
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS 정책 설정(필요시 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 헬스 체크 엔드포인트
@app.get("/health")
def health():
    # 간단한 상태 반환
    return {"ok": True}


# v1 라우터 등록 (prefix: /api/v1)
app.include_router(ollama_router.router, prefix="/api/v1")
app.include_router(rag_router.router, prefix="/api/v1")
app.include_router(mcp_router.router, prefix="/api/v1")
app.include_router(figma_router.router, prefix="/api/v1")
app.include_router(rag_faiss_router.router, prefix="/api/v1")

# (옵션) 이 파일을 직접 실행할 때만 uvicorn으로 기동
if __name__ == "__main__":
    # uvicorn 임포트
    import uvicorn

    # 개발용 실행 (reload는 모듈 재로드하므로, 프로덕션에선 비권장)
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
