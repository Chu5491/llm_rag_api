# app/api/v1/endpoints/rag.py

from typing import List
import json, re

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
    - 선택된 청크들을 RAG_BATCH_SIZE만큼 끊어서 여러 번 LLM 호출.
    """
    settings = get_settings()

    # 0) 인덱스 준비 확인
    if rag_vector_store.index is None:
        raise HTTPException(status_code=503, detail="벡터 인덱스가 아직 준비되지 않았습니다.")

    # 1) 검색용 쿼리는 .env에서 가져오기 (요청 바디의 query는 무시)
    search_query = settings.RAG_DEFAULT_QUERY

    # top_k / batch_size 설정
    search_top_k = settings.RAG_TOP_K
    rag_batch_size = settings.RAG_BATCH_SIZE

    total_chunks = len(rag_vector_store.texts)
    if total_chunks == 0:
        raise HTTPException(status_code=404, detail="인덱스에 저장된 문서 청크가 없습니다.")

    # top_k: 0 이하 → 전체 사용, 아니면 상위 N개
    if search_top_k and search_top_k > 0:
        top_k = min(search_top_k, total_chunks)
    else:
        top_k = total_chunks  # 전체 사용

    # 2) RAG 벡터 검색 (전체 랭킹 뽑고, 그중 상위 top_k만 사용)
    all_ranked = rag_vector_store.search(search_query, top_k=total_chunks)
    contexts = all_ranked[:top_k]

    if not contexts:
        raise HTTPException(status_code=404, detail="관련 문서를 찾지 못했습니다.")

    # batch_size가 0 이하거나 top_k보다 크면 → 한 번에 모두 보내기
    if rag_batch_size <= 0 or rag_batch_size >= len(contexts):
        rag_batch_size = len(contexts)

    # 3) 테스트케이스 개수 / ID prefix
    n = settings.RAG_TC_COUNT
    id_prefix = settings.RAG_TC_ID_PREFIX

    # 전체 contexts를 batch_size 만큼 잘라서 여러 번 LLM 호출
    def chunked(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]

    model_name = body.model or settings.LLM_MODEL
    all_testcases = []
    global_index = 1

    try:
        async with httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        ) as client:
            for batch_idx, batch in enumerate(chunked(contexts, rag_batch_size), start=1):
                # 3-1) 이 배치에 대한 CONTEXT 텍스트 구성
                context_text_parts: List[str] = []
                for i, item in enumerate(batch, start=1):
                    meta = item.get("meta", {})
                    source = meta.get("source", "unknown")
                    page = meta.get("page", None)

                    header = f"[{i}] source={source}"
                    if page is not None:
                        header += f", page={page}"

                    context_text_parts.append(f"{header}\n{item['text']}\n")

                context_text = "\n\n".join(context_text_parts)

                # 3-2) 이 배치 전용 규칙 프롬프트
                rules = (
                    f"당신은 QA 테스트케이스 전문가입니다.\n"
                    f"아래 규칙을 절대 어기지 말고, 오직 JSON 배열 하나만 출력하세요.\n"
                    f"- 출력: JSON 배열만 출력하고, 설명/해설 문장, 주석, 코드블록, 프로그래밍 코드를 절대 추가하지 마세요.\n"
                    f"- 'test_cases_list', 'test_cases_dict', 'print(...)' 같은 코드나 "
                    f"'#', '//', '/* */' 형태의 주석도 출력하지 마세요.\n"
                    f"- ```json, ```python 등의 마크다운 코드블록 표기는 절대 사용하지 마세요.\n"
                    f"- 최상위 구조: 테스트케이스 JSON 객체들의 배열이어야 합니다.\n"
                    f"- 각 원소는 아래 형식의 JSON **객체**여야 합니다:\n"
                    f'  {{\n'
                    f'    "testcase_id": "{id_prefix}_001",\n'
                    f'    "title": "제목",\n'
                    f'    "preconditions": "사전 조건",\n'
                    f'    "steps": ["1. ...", "2. ..."],\n'
                    f'    "expected_result": "예상 결과",\n'
                    f'    "priority": "High"\n'
                    f'  }}\n'
                    f'- 절대 다음과 같은 **배열 형태**로 출력하지 마세요 (잘못된 예):\n'
                    f'  ["{id_prefix}_001", "제목", "사전 조건", ["1. ..."], "예상 결과", "High"]\n'
                    f"- 객체 수: 가능한 한 {n}개에 가깝게, 최대 {n}개까지 생성하세요.\n"
                    f'- testcase_id: "{id_prefix}_001" 형식 사용. 첫 객체는 "{id_prefix}_001"부터 시작하고 이후 숫자를 1씩 증가시켜 사용합니다 '
                    f'(예: "{id_prefix}_001", "{id_prefix}_002", "{id_prefix}_003" … / 중간 번호 누락·중복 금지).\n'
                    f"- 필드: testcase_id, title, preconditions, steps, expected_result, priority\n"
                    f'- steps: 문자열 배열. 각 원소는 "1. ..." 형식으로 번호와 함께 한 줄로 작성합니다.\n'
                    f"- 각 텍스트 값은 80자 이내로 작성하고, 줄바꿈 문자(\\n)를 포함하지 마세요.\n"
                    f'- priority: \"High\", \"Medium\", \"Low\" 중 하나의 문자열 값으로 작성하세요.\n'
                    f"- 모든 텍스트 필드 값은 한국어로 작성하세요.\n"
                    f"중요: 아래 CONTEXT만 근거로 작성하고, CONTEXT에 없는 내용은 추측하여 추가하지 마세요. "
                    f"출력은 한국어 JSON 객체들의 배열만 포함해야 합니다.\n\n"
                )


                full_prompt = (
                    f"{rules}"
                    "### CONTEXT\n"
                    f"{context_text}\n"
                    "### END CONTEXT\n"
                )

                print(f"===== BATCH {batch_idx} FULL PROMPT BEGIN =====")
                print(full_prompt)
                print(f"===== BATCH {batch_idx} FULL PROMPT END =====")

                # 3-3) 이 배치에 대해 Ollama /api/generate 호출
                try:
                    resp = await client.post(
                        "/api/generate",
                        json={
                            "model": model_name,
                            "prompt": full_prompt,
                            "stream": False,
                        },
                    )
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Ollama 호출 실패 (batch {batch_idx}): {e}")

                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama 응답 오류 (batch {batch_idx}): {resp.status_code} {resp.text}",
                    )

                data = resp.json()
                content = (data.get("response") or "").strip()
                if not content:
                    # 그냥 이 배치는 스킵 (혹은 에러로 처리해도 됨)
                    continue
                
                print("===== START RAW RESPONSE =====")
                print(f"[batch {batch_idx}] raw LLM response:")
                print(content)
                print("===== END RAW RESPONSE =====")

                # 3-4) 이 배치의 JSON 파싱
                try:
                    raw_content = (data.get("response") or "").strip()
                    json_str = extract_json_array(raw_content)
                    batch_cases = json.loads(json_str)
                except json.JSONDecodeError:
                    
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM JSON 파싱 실패 (batch {batch_idx})",
                    )

                if not isinstance(batch_cases, list):
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM 응답이 JSON 배열 형식이 아닙니다 (batch {batch_idx})",
                    )

                # 3-5) 전역 testcase_id 재부여 + 전체 리스트에 합치기
                for tc in batch_cases:
                    # 필요한 필드만 믿고 나머지는 그대로 둠
                    tc["testcase_id"] = f"{id_prefix}_{global_index:03d}"
                    global_index += 1
                    all_testcases.append(tc)

    except HTTPException:
        # 위에서 던진 HTTPException은 그대로 전파
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 테스트케이스 생성 중 오류 발생: {e}")

    if not all_testcases:
        raise HTTPException(status_code=500, detail="LLM이 유효한 테스트케이스를 생성하지 못했습니다.")

    # 최종 answer는 JSON 배열 문자열
    answer_str = json.dumps(all_testcases, ensure_ascii=False)

    return RagQAResponse(
        answer=answer_str,
        contexts=contexts,
    )


def extract_json_array(text: str) -> str:
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        return match.group(0).strip()
    return text.strip()