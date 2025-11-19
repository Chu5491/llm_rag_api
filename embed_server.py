# -*- coding: utf-8 -*-
# 각 줄 위에 주석 스타일, 탭 들여쓰기 유지

# FastAPI 및 응답 타입 임포트
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 모델/타이핑
from pydantic import BaseModel
from typing import List, Optional

# 표준 라이브러리
import time, uuid, logging, json, os

# ML 관련
import torch
from sentence_transformers.SentenceTransformer import SentenceTransformer

# gzip 미들웨어 (가능하면 무조건 활성화)
try:
	# Starlette의 GZipMiddleware
	from starlette.middleware.gzip import GZipMiddleware
	_HAS_GZIP = True
except Exception:
	_HAS_GZIP = False

# 애플리케이션 인스턴스
app = FastAPI()

# 로거 설정 (필요한 정보만 INFO로)
logger = logging.getLogger("embed-server")
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

# 기본 모델명
DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"

# 모델 캐시
_models = {}

# 디바이스 선택
_device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로더
def get_model(model_id: Optional[str]):
	"""
	임베딩 모델 로드/캐시
	"""
	# 모델명 결정
	name = model_id or DEFAULT_MODEL
	# 캐시에 없으면 로드
	if name not in _models:
		t0 = time.perf_counter()
		logger.info(f"Loading model: {name} on {_device} ...")
		_models[name] = SentenceTransformer(name, device=_device)
		logger.info(f"Model loaded: {name} (took {(time.perf_counter()-t0)*1000:.1f} ms)")
	# 모델과 최종명 반환
	return _models[name], name

# 요청 바디 스키마
class EmbeddingRequest(BaseModel):
	# 사용할 모델 (옵션)
	model: Optional[str] = None
	# 입력 텍스트 배열
	input: List[str]
	# 정규화 여부
	normalize: Optional[bool] = True
	# 배치 사이즈
	batch_size: Optional[int] = 16
	# 토큰 길이 제한(참고용, 현재 미사용)
	max_length: Optional[int] = 512

# gzip 미들웨어 항상 활성(가능 시)
if _HAS_GZIP:
	# 최소 1KB 이상이면 gzip
	app.add_middleware(GZipMiddleware, minimum_size=1024)
	logger.info("GZipMiddleware enabled (minimum_size=1024)")
else:
	logger.info("GZipMiddleware not available; running without gzip")

# 스타트업 로그
@app.on_event("startup")
def _startup():
	"""
	서버 시작 시 정보 로그
	"""
	logger.info(f"Startup: device={_device}, default_model={DEFAULT_MODEL}")

# 헬스체크
@app.get("/health")
def health():
	"""
	헬스 체크 엔드포인트
	"""
	return {"ok": True, "device": _device, "default_model": DEFAULT_MODEL, "gzip": _HAS_GZIP}

# 임베딩 엔드포인트
@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest, request: Request):
	"""
	임베딩 생성 엔드포인트
	- 입력: EmbeddingRequest(JSON)
	- 출력: OpenAI 호환 형태의 임베딩 JSON
	"""
	# 요청 ID
	rid = request.headers.get("x-request-id", str(uuid.uuid4()))

	# 모델 로드
	model, resolved = get_model(req.model)

	# 시작 로그(요약)
	logger.info(f"rid={rid} start model='{resolved}' n={len(req.input)} batch={req.batch_size} norm={req.normalize}")

	# 인코딩 수행
	t0 = time.perf_counter()
	vecs = model.encode(
		req.input,
		batch_size=req.batch_size,
		convert_to_numpy=True,
		normalize_embeddings=req.normalize,
		show_progress_bar=False
	)
	elapsed_ms = (time.perf_counter() - t0) * 1000

	# 차원 계산
	if hasattr(vecs, "shape") and len(vecs.shape) == 2:
		dim = int(vecs.shape[1])
	else:
		dim = len(vecs[0]) if vecs else 0

	# 응답 페이로드 구성
	data = [{"object": "embedding", "index": i, "embedding": vec.tolist()} for i, vec in enumerate(vecs)]
	payload = {"object": "list", "model": resolved, "data": data}

	# 완료 로그(요약)
	logger.info(f"rid={rid} done dim={dim} took={elapsed_ms:.1f}ms")

	# JSONResponse 반환 (gzip 미들웨어가 있으면 자동 압축/전송)
	return JSONResponse(payload)

# 로컬 실행 진입점
if __name__ == "__main__":
	import uvicorn
	# uvicorn 실행 (HTTP/1.1 고정, keep-alive 짧게)
	uvicorn.run(
		app,
		host="0.0.0.0",
		port=8888,
		log_level="info",
		http="h11",
		ws="none",
		timeout_keep_alive=5
	)
