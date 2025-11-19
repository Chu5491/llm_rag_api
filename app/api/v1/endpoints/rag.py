# app/api/v1/endpoints/rag.py

from typing import List

# FastAPI 라우터
from fastapi import APIRouter

# 방금 만든 EmbeddingService 사용
from app.services.embeddings import embedding_service

# httpx 비동기 클라이언트
import httpx
from fastapi import APIRouter, Query, HTTPException

# 설정 가져오기
from app.core.config import get_settings

# RAG 스키마
from app.services.rag_store import rag_vector_store
from app.schemas.rag import EmbedDebugResponse, RagQARequest, RagQAResponse

# /api/v1/rag 아래로 묶이는 라우터
router = APIRouter(prefix="/rag", tags=["rag"])

@router.get("/embed-debug", response_model=EmbedDebugResponse)
async def embed_debug():
	"""임베딩 서비스가 정상 동작하는지 확인하는 엔드포인트"""
	# 테스트용 문장 하나
	sample = "RAG 시스템 임베딩 디버그 입니다."

	# 쿼리 임베딩 계산
	vec = embedding_service.embed_query(sample)
	# vec shape: (1, dim) → 길이는 두 번째 축 크기
	vector_length = int(vec.shape[1])

	# 디바이스와 차원 정보 반환
	return EmbedDebugResponse(
		device=embedding_service.device,
		dimension=embedding_service.dimension,
		vector_length=vector_length,
	)

@router.post("/qa", response_model=RagQAResponse)
async def rag_qa(body: RagQARequest) -> RagQAResponse:
	"""
	업로드된 문서들을 기반으로 RAG QA를 수행하는 엔드포인트

	1) FAISS에서 관련 청크 top_k개 검색
	2) 그 청크들을 context로 묶어서
	3) Ollama LLM에 전달해 답변 생성
	"""
	settings = get_settings()

	# 인덱스 준비 확인
	if rag_vector_store.index is None:
		raise HTTPException(status_code=503, detail="벡터 인덱스가 아직 준비되지 않았습니다.")

	# 1) 벡터 검색
	contexts = rag_vector_store.search(query=body.query, top_k=body.top_k)
	if not contexts:
		raise HTTPException(status_code=404, detail="관련 문서를 찾지 못했습니다.")

	# 2) 컨텍스트 텍스트 구성
	context_text_parts: List[str] = []
	for i, item in enumerate(contexts, start=1):
		meta = item.get("meta", {})
		source = meta.get("source", "unknown")
		page = meta.get("page", None)

		header = f"[{i}] source={source}"
		if page is not None:
			header += f", page={page}"

		context_text_parts.append(f"{header}\n{item['text']}\n")

	context_text = "\n\n".join(context_text_parts)

	system_prompt = (
		"너는 업로드된 문서 내용을 기반으로 질문에 답하는 한국어 어시스턴트야. "
		"반드시 제공된 컨텍스트 안에서만 답변하고, 모르는 내용은 모른다고 말해."
	)

	user_prompt = (
		f"다음은 참고용 문서 일부야:\n\n"
		f"{context_text}\n\n"
		f"위 자료만 근거로 아래 질문에 한국어로 자세히 답변해 줘.\n"
		f"질문: {body.query}"
	)

	model_name = body.model or settings.LLM_MODEL

	# 3) Ollama /api/chat 호출
	try:
		async with httpx.AsyncClient(
			base_url=settings.OLLAMA_BASE_URL,
			timeout=settings.OLLAMA_TIMEOUT,
		) as client:
			resp = await client.post(
				"/api/chat",
				json={
					"model": model_name,
					"messages": [
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_prompt},
					],
					"stream": False,
				},
			)
	except Exception as e:
		raise HTTPException(status_code=502, detail=f"Ollama 호출 실패: {e}")

	if resp.status_code != 200:
		raise HTTPException(
			status_code=502,
			detail=f"Ollama 응답 오류: {resp.status_code} {resp.text}",
		)

	data = resp.json()
	message = data.get("message") or {}
	answer = message.get("content", "").strip()

	if not answer:
		raise HTTPException(status_code=500, detail="LLM이 빈 응답을 반환했습니다.")

	# 4) 스키마에 맞춰 응답 생성
	return RagQAResponse(
		answer=answer,
		contexts=contexts,
	)
