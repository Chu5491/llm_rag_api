# app/api/v1/endpoints/rag.py
from typing import List
import json

# FastAPI 라우터
from fastapi import APIRouter

# 방금 만든 EmbeddingService 사용
from app.services.embeddings import embedding_service

# httpx 비동기 클라이언트
import httpx
from fastapi import HTTPException

# 설정 가져오기
from app.api.deps import get_app_settings

# RAG 스키마
from app.services.file_rag_store import file_rag_vector_store
from app.services.figma_rag_store import figma_rag_vector_store
from app.schemas.rag import EmbedDebugResponse, RagQARequest, RagQAResponse

# 로거
from app.core.logging import get_logger

# Parser
from app.utils.rag_parser import extract_json_array


logger = get_logger(__name__)
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


@router.post("/generate/file", response_model=RagQAResponse)
async def rag_generate(body: RagQARequest) -> RagQAResponse:
    """
    업로드된 PDF/문서를 기반으로
    QA 테스트케이스 JSON을 생성하는 일회성 RAG 엔드포인트.

    - 쿼리 텍스트는 .env(RAG_DEFAULT_QUERY)에서 가져온다.
    - top_k가 0 이하이면: 전체 컨텍스트 사용.
    - LLM 호출은 Ollama /api/generate 사용.
    - 선택된 컨텍스트들을 RAG_BATCH_SIZE만큼 끊어서 여러 번 LLM 호출.
    """
    settings = get_app_settings()

    # 0) 인덱스 준비 확인
    if file_rag_vector_store.index is None:
        raise HTTPException(
            status_code=503, detail="벡터 인덱스가 아직 준비되지 않았습니다."
        )

    # 1) 검색용 쿼리는 .env에서 가져오기 (요청 바디의 query는 무시)
    search_query = settings.RAG_DEFAULT_QUERY

    # top_k / batch_size 설정
    search_top_k = settings.RAG_TOP_K
    rag_batch_size = settings.RAG_BATCH_SIZE

    total_items = len(file_rag_vector_store.texts)
    if total_items == 0:
        raise HTTPException(status_code=404, detail="인덱스에 저장된 문서가 없습니다.")

    # top_k: 0 이하 → 전체 사용, 아니면 상위 N개
    if search_top_k and search_top_k > 0:
        top_k = min(search_top_k, total_items)
    else:
        top_k = total_items  # 전체 사용

    logger.info(
        "[RAG-File] 전체 컨텍스트 수=%d, 설정 top_k=%s → 실제 사용 컨텍스트 수=%d",
        total_items,
        search_top_k,
        top_k,
    )

    # 2) RAG 벡터 검색 (전체 랭킹 뽑고, 그중 상위 top_k만 사용)
    all_ranked = file_rag_vector_store.search(search_query, top_k=total_items)
    contexts = all_ranked[:top_k]

    if not contexts:
        raise HTTPException(status_code=404, detail="관련 문서를 찾지 못했습니다.")

    # batch_size가 0 이하거나 top_k보다 크면 → 한 번에 모두 보내기
    if rag_batch_size <= 0 or rag_batch_size >= len(contexts):
        rag_batch_size = len(contexts)

    # 총 배치 수 계산
    num_batches = (len(contexts) + rag_batch_size - 1) // rag_batch_size
    logger.info(
        "[RAG-File] 배치 실행 준비 - 전체 컨텍스트=%d, 배치 크기=%d, 총 배치 수=%d",
        len(contexts),
        rag_batch_size,
        num_batches,
    )

    # 3) 테스트케이스 개수 / ID prefix
    n = settings.RAG_TC_COUNT
    id_prefix = settings.RAG_TC_ID_PREFIX
    model_name = body.model or settings.LLM_MODEL
    logger.info(
        "[RAG-File] 요청 테스트케이스 개수 n=%d, ID prefix=%s, model=%s",
        n,
        id_prefix,
        model_name,
    )

    # 전체 contexts를 batch_size 만큼 잘라서 여러 번 LLM 호출
    def chunked(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    all_testcases = []
    global_index = 1

    try:
        async with httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        ) as client:
            for batch_idx, batch in enumerate(
                chunked(contexts, rag_batch_size), start=1
            ):
                remaining_batches = num_batches - batch_idx
                logger.info(
                    "[RAG-File] Batch %d/%d 시작 - 현재 배치 크기=%d, 남은 배치 수=%d",
                    batch_idx,
                    num_batches,
                    len(batch),
                    remaining_batches,
                )

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
                logger.debug(
                    "[RAG-File] Batch %d CONTEXT 구성 완료 - 길이=%d chars",
                    batch_idx,
                    len(context_text),
                )

                # 3-2) 이 배치 전용 규칙 프롬프트
                rules = (
                    f"당신은 QA 전문가입니다. CONTEXT를 분석하여 유의미한 테스트케이스를 선별해 작성하세요.\n"
                    f"\n"
                    f"## JSON 출력 형식 (배열)\n"
                    f"- 오직 JSON 배열만 출력 (마크다운/설명 금지)\n"
                    f"- 객체 스키마:\n"
                    f"  {{\n"
                    f'    "testcase_id": "{id_prefix}_001",\n'
                    f'    "module": "기능/모듈명(예: 회원관리, 주문)",\n'
                    f'    "title": "테스트 제목(80자 이내)",\n'
                    f'    "preconditions": "사전 조건",\n'
                    f'    "steps": ["1. 동작", "2. 확인"],\n'
                    f'    "expected_result": "예상 결과",\n'
                    f'    "priority": "High|Medium|Low"\n'
                    f"  }}\n"
                    f"- ID는 {id_prefix}_001부터 순차 증가.\n"
                    f"\n"
                    f"## 핵심 규칙\n"
                    f"1. **최대 {n}개 제한**: 의미 있는 테스트만 선별하여 생성. 억지로 {n}개를 채우지 말고 필요한 만큼만 작성.\n"
                    f"2. **Context 기반**: 문서/화면에 명시된 기능만 검증. 언급 없는 기능(로그인, 결제 등)은 절대 상상하여 추가 금지.\n"
                    f"3. **Module 그룹화**: 유사한 기능이나 화면 단위로 'module' 필드를 통일하여 작성.\n"
                    f"4. **내용 작성**: 한국어로 작성하되, UI 텍스트/API 경로는 원문 유지.\n"
                )

                full_prompt = f"{rules}### CONTEXT\n{context_text}\n### END CONTEXT\n"

                logger.info(
                    "[RAG-File] Batch %d/%d Ollama 호출 시작 - model=%s",
                    batch_idx,
                    num_batches,
                    model_name,
                )

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
                    logger.error(
                        "[RAG-File] Batch %d Ollama 호출 실패: %s", batch_idx, e
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama 호출 실패 (batch {batch_idx}): {e}",
                    )

                logger.info(
                    "[RAG-File] Batch %d/%d Ollama 응답 수신 - status=%d",
                    batch_idx,
                    num_batches,
                    resp.status_code,
                )

                if resp.status_code != 200:
                    logger.error(
                        "[RAG-File] Batch %d Ollama 응답 오류: %d %s",
                        batch_idx,
                        resp.status_code,
                        resp.text,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama 응답 오류 (batch {batch_idx}): {resp.status_code} {resp.text}",
                    )

                data = resp.json()
                content = (data.get("response") or "").strip()
                if not content:
                    logger.warning(
                        "[RAG-File] Batch %d 응답 내용 없음, 배치 스킵", batch_idx
                    )
                    continue

                # 3-4) 이 배치의 JSON 파싱
                try:
                    raw_content = (data.get("response") or "").strip()
                    json_str = extract_json_array(raw_content)
                    batch_cases = json.loads(json_str)
                    logger.info(
                        "[RAG-File] Batch %d 테스트케이스 JSON 파싱 성공 - 개수=%d",
                        batch_idx,
                        len(batch_cases),
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        "[RAG-File] Batch %d JSON 파싱 실패: %s / raw=%s",
                        batch_idx,
                        e,
                        raw_content[:200],
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM JSON 파싱 실패 (batch {batch_idx})",
                    )

                if not isinstance(batch_cases, list):
                    logger.error(
                        "[RAG-File] Batch %d 응답 루트가 배열이 아님: type=%s",
                        batch_idx,
                        type(batch_cases),
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM 응답이 JSON 배열 형식이 아닙니다 (batch {batch_idx})",
                    )

                # 3-5) 전역 testcase_id 재부여 + 전체 리스트에 합치기
                for tc in batch_cases:
                    tc["testcase_id"] = f"{id_prefix}_{global_index:03d}"
                    global_index += 1
                    all_testcases.append(tc)

                remaining_batches = num_batches - batch_idx
                logger.info(
                    "[RAG-File] Batch %d/%d 처리 완료 - 누적 테스트케이스 수=%d, 남은 배치 수=%d",
                    batch_idx,
                    num_batches,
                    len(all_testcases),
                    remaining_batches,
                )

    except HTTPException:
        # 위에서 던진 HTTPException은 그대로 전파
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"RAG 테스트케이스 생성 중 오류 발생: {e}"
        )

    if not all_testcases:
        raise HTTPException(
            status_code=500, detail="LLM이 유효한 테스트케이스를 생성하지 못했습니다."
        )

    # 최종 answer는 JSON 배열 문자열
    answer_str = json.dumps(all_testcases, ensure_ascii=False)

    logger.info(
        "[RAG-File] 최종 테스트케이스 생성 완료 - 총 %d개, 사용 배치 수=%d",
        len(all_testcases),
        num_batches,
    )

    return RagQAResponse(
        answer=answer_str,
        contexts=contexts,
    )


@router.post("/generate/figma", response_model=RagQAResponse)
async def rag_generate_from_figma(body: RagQARequest) -> RagQAResponse:
    """
    Figma에서 추출/인덱싱된 화면 요약(FigmaRAG)을 기반으로
    QA 테스트케이스 JSON을 생성하는 일회성 RAG 엔드포인트.

    - 쿼리 텍스트는 .env(RAG_DEFAULT_QUERY)에서 가져온다.
    - top_k가 0 이하이면: 전체 컨텍스트 사용.
    - LLM 호출은 Ollama /api/generate 사용.
    - 선택된 컨텍스트들을 RAG_BATCH_SIZE만큼 끊어서 여러 번 LLM 호출.
    """
    settings = get_app_settings()
    logger.info("[Figma-RAG] /generate/figma 요청 수신")

    # 0) Figma RAG 기능 활성 여부 확인
    if not figma_rag_vector_store.enabled:
        logger.error("[Figma-RAG] FIGMA_ENABLED = False, 기능 비활성 상태")
        raise HTTPException(
            status_code=503,
            detail="Figma RAG 기능이 비활성화되어 있습니다. FIGMA_ENABLED 설정을 확인하세요.",
        )

    # 0-1) 인덱스 준비 확인
    if figma_rag_vector_store.index is None:
        logger.error("[Figma-RAG] Figma 벡터 인덱스가 초기화되지 않음")
        raise HTTPException(
            status_code=503, detail="Figma 벡터 인덱스가 아직 준비되지 않았습니다."
        )

    # 1) 검색용 쿼리는 .env에서 가져오기 (요청 바디의 query는 무시)
    search_query = settings.RAG_DEFAULT_QUERY
    logger.info(f"[Figma-RAG] 검색 쿼리 사용: {search_query!r}")

    # top_k / batch_size 설정
    search_top_k = settings.RAG_TOP_K
    rag_batch_size = settings.RAG_BATCH_SIZE

    total_items = len(figma_rag_vector_store.texts)
    logger.info(f"[Figma-RAG] 인덱스 내 Figma 컨텍스트 수: {total_items}")

    if total_items == 0:
        logger.warning("[Figma-RAG] 인덱스에 저장된 Figma 컨텍스트가 없음")
        raise HTTPException(
            status_code=404, detail="Figma 인덱스에 저장된 컨텍스트가 없습니다."
        )

    # top_k: 0 이하 → 전체 사용, 아니면 상위 N개
    if search_top_k and search_top_k > 0:
        top_k = min(search_top_k, total_items)
    else:
        top_k = total_items  # 전체 사용
    logger.info(
        "[Figma-RAG] 전체 컨텍스트 수=%d, 설정 top_k=%s → 실제 사용 컨텍스트 수=%d",
        total_items,
        search_top_k,
        top_k,
    )

    # 2) RAG 벡터 검색 (전체 랭킹 뽑고, 그중 상위 top_k만 사용)
    logger.info("[Figma-RAG] Figma 벡터 검색 시작 (full ranking)")
    all_ranked = figma_rag_vector_store.search(search_query, top_k=total_items)
    contexts = all_ranked[:top_k]
    logger.info(f"[Figma-RAG] 검색 결과 상위 {len(contexts)}개 컨텍스트 선택")

    if not contexts:
        logger.warning("[Figma-RAG] 관련 Figma 화면 검색 결과 없음")
        raise HTTPException(
            status_code=404, detail="관련 Figma 화면을 찾지 못했습니다."
        )

    # batch_size가 0 이하거나 top_k보다 크면 → 한 번에 모두 보내기
    if rag_batch_size <= 0 or rag_batch_size >= len(contexts):
        rag_batch_size = len(contexts)

    num_batches = (len(contexts) + rag_batch_size - 1) // rag_batch_size
    logger.info(
        "[Figma-RAG] 배치 실행 준비 - 전체 컨텍스트=%d, 배치 크기=%d, 총 배치 수=%d",
        len(contexts),
        rag_batch_size,
        num_batches,
    )

    # 3) 테스트케이스 개수 / ID prefix
    n = settings.RAG_TC_COUNT
    id_prefix = settings.RAG_TC_ID_PREFIX
    model_name = body.model or settings.LLM_MODEL
    logger.info(
        "[Figma-RAG] 요청 테스트케이스 개수 n=%d, ID prefix=%s, model=%s",
        n,
        id_prefix,
        model_name,
    )

    # 전체 contexts를 batch_size 만큼 잘라서 여러 번 LLM 호출
    def chunked(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    all_testcases = []
    global_index = 1

    try:
        async with httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        ) as client:
            for batch_idx, batch in enumerate(
                chunked(contexts, rag_batch_size), start=1
            ):
                remaining_batches = num_batches - batch_idx
                logger.info(
                    "[Figma-RAG] Batch %d/%d 시작 - 현재 배치 크기=%d, 남은 배치 수=%d",
                    batch_idx,
                    num_batches,
                    len(batch),
                    remaining_batches,
                )

                # 3-1) 이 배치에 대한 CONTEXT 텍스트 구성
                context_text_parts: List[str] = []
                for i, item in enumerate(batch, start=1):
                    meta = item.get("meta", {})
                    source = meta.get("source", "figma")
                    page_name = meta.get("page_name")
                    variant = meta.get("variant")
                    section_name = meta.get("section_name")
                    screen_path = meta.get("screen_path")

                    # Figma 전용 헤더 구성
                    header = f"[{i}] source={source}"
                    if page_name:
                        header += f", page={page_name}"
                    if variant:
                        header += f", variant={variant}"
                    if section_name:
                        header += f", section={section_name}"
                    if screen_path:
                        header += f", path={screen_path}"

                    context_text_parts.append(f"{header}\n{item['text']}\n")

                context_text = "\n\n".join(context_text_parts)
                logger.debug(
                    "[Figma-RAG] Batch %d CONTEXT 구성 완료 - 길이=%d chars",
                    batch_idx,
                    len(context_text),
                )

                # 3-2) 이 배치 전용 규칙 프롬프트 (파일용과 동일, 재사용)
                rules = (
                    f"당신은 QA 전문가입니다. Figma 화면 정보를 분석하여 **사용자 행동(Action) 중심**의 테스트케이스를 작성하세요.\n"
                    f"\n"
                    f"## JSON 출력 형식 (배열)\n"
                    f"- 오직 JSON 배열만 출력 (마크다운/설명 금지)\n"
                    f"- 객체 스키마:\n"
                    f"  {{\n"
                    f'    "testcase_id": "{id_prefix}_001",\n'
                    f'    "module": "화면/기능명(예: 메인홈, 설정)",\n'
                    f'    "title": "테스트 제목(80자 이내)",\n'
                    f'    "preconditions": "사전 조건",\n'
                    f'    "steps": ["1. [행동] 버튼 클릭...", "2. [확인] 화면 이동..."],\n'
                    f'    "expected_result": "예상 결과",\n'
                    f'    "priority": "High|Medium|Low"\n'
                    f"  }}\n"
                    f"- ID는 {id_prefix}_001부터 순차 증가.\n"
                    f"\n"
                    f"## 핵심 규칙 (위반 시 실패)\n"
                    f"1. **단순 텍스트/스타일 확인 절대 금지**: \n"
                    f"   - 'XXX 텍스트가 존재하는지 확인', '폰트 크기 확인', '아이콘 존재 확인' 같은 정적 검증용 TC는 **생성하지 마세요**.\n"
                    f"   - 텍스트 확인은 기능/화면 이동 테스트의 'expected_result'에 부가적으로만 포함하세요.\n"
                    f"2. **User Action 필수**: \n"
                    f"   - 모든 TC는 클릭, 탭, 입력, 스크롤, 전환 등 **사용자의 능동적 행동**을 포함해야 합니다.\n"
                    f"   - 예: '버튼 클릭 → 모달 오픈', '탭 전환 → 컨텐츠 변경', '메뉴 클릭 → 페이지 이동'.\n"
                    f"3. **Context 기반**: Figma 데이터에 명시된 버튼/링크/인터랙션만 검증. 없는 로직(실제 결제, 로그인 처리 등) 상상 금지.\n"
                    f"4. **Module 그룹화**: 화면이나 섹션 단위로 'module' 필드를 통일하여 작성.\n"
                    f"5. **최대 {n}개**: 의미 있는 인터랙션 테스트만 선별.\n"
                )

                full_prompt = f"{rules}### CONTEXT\n{context_text}\n### END CONTEXT\n"
                logger.info(
                    "[Figma-RAG] Batch %d/%d Ollama 호출 시작 - model=%s",
                    batch_idx,
                    num_batches,
                    model_name,
                )

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
                    logger.error(
                        "[Figma-RAG] Batch %d Ollama 호출 실패: %s", batch_idx, e
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama 호출 실패 (batch {batch_idx}): {e}",
                    )

                logger.info(
                    "[Figma-RAG] Batch %d/%d Ollama 응답 수신 - status=%d",
                    batch_idx,
                    num_batches,
                    resp.status_code,
                )

                if resp.status_code != 200:
                    logger.error(
                        "[Figma-RAG] Batch %d Ollama 응답 오류: %d %s",
                        batch_idx,
                        resp.status_code,
                        resp.text,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama 응답 오류 (batch {batch_idx}): {resp.status_code} {resp.text}",
                    )

                data = resp.json()
                content = (data.get("response") or "").strip()
                if not content:
                    logger.warning(
                        "[Figma-RAG] Batch %d 응답 내용 없음, 배치 스킵", batch_idx
                    )
                    continue

                # 3-4) 이 배치의 JSON 파싱
                try:
                    raw_content = (data.get("response") or "").strip()
                    json_str = extract_json_array(raw_content)
                    batch_cases = json.loads(json_str)
                    logger.info(
                        "[Figma-RAG] Batch %d 테스트케이스 JSON 파싱 성공 - 개수=%d",
                        batch_idx,
                        len(batch_cases),
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        "[Figma-RAG] Batch %d JSON 파싱 실패: %s / raw=%s",
                        batch_idx,
                        e,
                        raw_content[:200],
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM JSON 파싱 실패 (batch {batch_idx})",
                    )

                if not isinstance(batch_cases, list):
                    logger.error(
                        "[Figma-RAG] Batch %d 응답 루트가 배열이 아님: type=%s",
                        batch_idx,
                        type(batch_cases),
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM 응답이 JSON 배열 형식이 아닙니다 (batch {batch_idx})",
                    )

                # 3-5) 전역 testcase_id 재부여 + 전체 리스트에 합치기
                for tc in batch_cases:
                    tc["testcase_id"] = f"{id_prefix}_{global_index:03d}"
                    global_index += 1
                    all_testcases.append(tc)

                remaining_batches = num_batches - batch_idx
                logger.info(
                    "[Figma-RAG] Batch %d/%d 처리 완료 - 누적 테스트케이스 수=%d, 남은 배치 수=%d",
                    batch_idx,
                    num_batches,
                    len(all_testcases),
                    remaining_batches,
                )

    except HTTPException:
        # 위에서 던진 HTTPException은 그대로 전파
        logger.exception("[Figma-RAG] HTTPException 발생, 상위로 전파")
        raise
    except Exception as e:
        logger.exception("[Figma-RAG] 알 수 없는 예외 발생")
        raise HTTPException(
            status_code=500, detail=f"Figma RAG 테스트케이스 생성 중 오류 발생: {e}"
        )

    if not all_testcases:
        logger.error(
            "[Figma-RAG] LLM이 유효한 테스트케이스를 생성하지 못함 (all_testcases 비어 있음)"
        )
        raise HTTPException(
            status_code=500,
            detail="LLM이 유효한 테스트케이스를 생성하지 못했습니다.",
        )

    # 최종 answer는 JSON 배열 문자열
    answer_str = json.dumps(all_testcases, ensure_ascii=False)
    logger.info(
        "[Figma-RAG] 최종 테스트케이스 생성 완료 - 총 %d개, 사용 배치 수=%d",
        len(all_testcases),
        num_batches,
    )

    return RagQAResponse(
        answer=answer_str,
        contexts=contexts,
    )
