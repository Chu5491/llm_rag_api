# FastAPI 임포트
from fastapi import FastAPI
# GZip 미들웨어 임포트
from starlette.middleware.gzip import GZipMiddleware
# CORS 미들웨어 임포트
from fastapi.middleware.cors import CORSMiddleware
# 라우터 임포트
from app.api.v1.endpoints import ollama as ollama_router, rag as rag_router

# 애플리케이션 인스턴스 생성
app = FastAPI(title="RAG/Ollama API", version="0.1.0")

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

# (옵션) 이 파일을 직접 실행할 때만 uvicorn으로 기동
if __name__ == "__main__":
	# uvicorn 임포트
	import uvicorn
	# 개발용 실행 (reload는 모듈 재로드하므로, 프로덕션에선 비권장)
	uvicorn.run(
		"app.main:app",
		host="0.0.0.0",
		port=8000,
		reload=True,
		log_level="info"
	)
