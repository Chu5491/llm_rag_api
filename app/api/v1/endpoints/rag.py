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

@router.post("/generate", response_model=RagQAResponse)
async def rag_generate(body: RagQARequest) -> RagQAResponse:
	"""
	업로드된 PDF/문서를 기반으로
	QA 테스트케이스 JSON을 생성하는 일회성 RAG 엔드포인트.

	- 쿼리 텍스트는 .env(RAG_DEFAULT_QUERY)에서 가져온다.
	- top_k가 0 이하이면: 전체 청크 사용.
	- LLM 호출은 Ollama /api/generate 사용.
	"""
	settings = get_settings()

	# 0) 인덱스 준비 확인
	if rag_vector_store.index is None:
		raise HTTPException(status_code=503, detail="벡터 인덱스가 아직 준비되지 않았습니다.")

	# 1) 검색용 쿼리는 .env에서 가져오기 (요청 바디의 query는 무시)
	search_query = settings.RAG_DEFAULT_QUERY

	# top_k 설정:
	search_top_k = settings.RAG_TOP_K
	total_chunks = len(rag_vector_store.texts)
	if total_chunks == 0:
		raise HTTPException(status_code=404, detail="인덱스에 저장된 문서 청크가 없습니다.")

	if search_top_k and search_top_k > 0:
		top_k = min(search_top_k, total_chunks)
	else:
		top_k = total_chunks  # 전체 사용

	# 2) RAG 벡터 검색
	contexts = rag_vector_store.search(query=search_query, top_k=top_k)
	if not contexts:
		raise HTTPException(status_code=404, detail="관련 문서를 찾지 못했습니다.")

	# 3) CONTEXT 텍스트 구성
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

	# 4) 테스트케이스 규칙 프롬프트 구성
	n = settings.RAG_TC_COUNT
	id_prefix = settings.RAG_TC_ID_PREFIX
	testcase_ids = [f"{id_prefix}-{i+1:03d}" for i in range(n)]
	ids_str = ", ".join(f"\"{tc_id}\"" for tc_id in testcase_ids)

	rules = (
		f"당신은 QA 테스트케이스 전문가입니다.\n"
		f"규칙을 어기지 말고 JSON 배열 하나만 출력하세요.\n"
		f"- 객체 수: {n}\n"
		f"- testcase_id: [{ids_str}] 만 사용(누락/중복 금지)\n"
		f"- 필드: testcase_id, title, preconditions, steps, expected_result, priority\n"
		f"- steps: 문자열 배열(각 원소는 \"1. ...\" 형식)\n"
		f"- 각 텍스트는 80자 이내, 줄바꿈 금지\n"
		f"- priority: High|Medium|Low\n"
		f"- 공백 제외 '['로 시작, ']'로 끝나야 함\n"
		f"중요: 아래 CONTEXT만 근거로 작성(추측 금지). 출력은 한국어 JSON만.\n\n"
	)

	full_prompt = (
		f"{rules}"
		f"### CONTEXT\n"
		f"{context_text}\n"
		f"### END CONTEXT\n"
	)
	print("===== FULL PROMPT BEGIN =====")
	print(full_prompt)
	print("===== FULL PROMPT END =====")
	
	# 5) Ollama /api/generate 호출
	model_name = body.model or settings.LLM_MODEL

	try:
		async with httpx.AsyncClient(
			base_url=settings.OLLAMA_BASE_URL,
			timeout=settings.OLLAMA_TIMEOUT,
		) as client:
			resp = await client.post(
				"/api/generate",
				json={
					"model": model_name,
					"prompt": full_prompt,
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
	# /api/generate 응답 형식: { "response": "......", "done": true, ... }
	content = (data.get("response") or "").strip()

	if not content:
		raise HTTPException(status_code=500, detail="LLM이 빈 응답을 반환했습니다.")

	# RagQAResponse.answer 에는 "JSON 배열" 문자열이 그대로 들어감
	return RagQAResponse(
		answer=content,
		contexts=contexts,  # <- 오타 조심! 실제 코드에선 contexts
	)