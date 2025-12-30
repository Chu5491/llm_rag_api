# app/api/v1/endpoints/rag.py
from typing import List, Dict
import json
import asyncio

# FastAPI 라우터
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.database import get_vector_db
from app.crud.config import get_app_config
from app.crud.rag import search_all_sources
from app.crud.history import (
    create_history,
    update_progress,
    complete_history,
    get_history_by_id,
)

# 방금 만든 EmbeddingService 사용
from app.services.embeddings import embedding_service

# httpx 비동기 클라이언트
import httpx

# 설정 가져오기
from app.api.deps import get_app_settings

# RAG 서비스 (Postgres 버전)
from app.services.file_rag_store_pg import file_rag_vector_store_pg
from app.services.figma_rag_store_pg import figma_rag_vector_store_pg
from app.schemas.rag import EmbedDebugResponse, RagQARequest, RagQAResponse
from app.models.rag_embeddings import RagEmbedding

# 로거
from app.core.logging import get_logger

# Parser
from app.utils.rag_parser import extract_json_array


logger = get_logger(__name__)


# /api/v1/rag 아래로 묶이는 라우터
router = APIRouter(prefix="/rag", tags=["rag"])


# Global dictionary to track active tasks
_active_tasks: Dict[int, asyncio.Task] = {}


def get_active_tasks() -> Dict[int, asyncio.Task]:
    """Get the global _active_tasks dictionary"""
    global _active_tasks
    return _active_tasks


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
async def rag_generate(
    body: RagQARequest, db: Session = Depends(get_vector_db)
) -> RagQAResponse:
    """
    업로드된 PDF/문서를 기반으로
    QA 테스트케이스 JSON을 생성하는 일회성 RAG 엔드포인트. (Postgres 버전)

    - 쿼리 텍스트는 .env(RAG_DEFAULT_QUERY)에서 가져온다.
    - DB에서 통합 검색(source_type=file)을 수행한다.
    """
    settings = get_app_settings()
    config = get_app_config(db)
    if not config:
        raise HTTPException(
            status_code=404,
            detail="DB에 시스템 설정(Configurations)이 존재하지 않습니다.",
        )

    # 1) 검색용 쿼리는 .env에서 가져오기 (요청 바디의 query는 무시)
    search_query = settings.RAG_DEFAULT_QUERY

    # top_k / batch_size 설정
    search_top_k = settings.RAG_TOP_K

    # DB 설정 적용
    rag_batch_size = config.rag_batch_size
    n = config.rag_tc_count
    id_prefix = config.rag_tc_id_prefix
    llm_model = config.llm_model

    # DB에서 현재 소스에 해당하는 총 아이템 수 확인
    total_items = file_rag_vector_store_pg.get_count()
    if total_items == 0:
        logger.warning("[RAG-File-PG] DB에 데이터가 없음")
        raise HTTPException(status_code=404, detail="DB에 저장된 파일 문서가 없습니다.")

    # top_k: 0 이하 → 전체 사용, 아니면 상위 N개
    if search_top_k and search_top_k > 0:
        top_k = min(search_top_k, total_items)
    else:
        top_k = total_items  # 전체 사용

    logger.info(
        "[RAG-File-PG] 전체 컨텍스트 수=%d, 설정 top_k=%s → 실제 사용 컨텍스트 수=%d",
        total_items,
        search_top_k,
        top_k,
    )

    # 2) RAG 벡터 검색 (PG 전용 검색 로직 사용)
    contexts = file_rag_vector_store_pg.search(search_query, top_k=top_k)

    if not contexts:
        logger.warning(f"[RAG-File-PG] 검색 쿼리: {search_query} 에 대한 결과 없음")
        raise HTTPException(status_code=404, detail="관련 문서를 찾지 못했습니다.")

    # batch_size 설정
    if rag_batch_size <= 0 or rag_batch_size >= len(contexts):
        rag_batch_size = len(contexts)

    # 총 배치 수 계산
    num_batches = (len(contexts) + rag_batch_size - 1) // rag_batch_size
    logger.info(
        "[RAG-File-PG] 배치 실행 준비 - 전체 컨텍스트=%d, 배치 크기=%d, 총 배치 수=%d",
        len(contexts),
        rag_batch_size,
        num_batches,
    )

    # 3) 테스트케이스 개수 / ID prefix
    # n, id_prefix는 위에서 DB 설정으로 이미 가져옴
    model_name = body.model or llm_model
    logger.info(
        "[RAG-File-PG] 요청 n=%d, ID prefix=%s, model=%s",
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
                    "[RAG-File-PG] Batch %d/%d 시작 - 현재 배치 크기=%d, 남은 배치 수=%d",
                    batch_idx,
                    num_batches,
                    len(batch),
                    remaining_batches,
                )

                # 3-1) CONTEXT 텍스트 구성
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
                    f"[RAG-File-PG] Batch {batch_idx} CONTEXT 길이: {len(context_text)} chars"
                )

                # 3-2) 프롬프트 규칙 복구
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

                # 3-3) Ollama 호출
                resp = await client.post(
                    "/api/generate",
                    json={
                        "model": model_name,
                        "prompt": full_prompt,
                        "stream": False,
                    },
                )

                if resp.status_code != 200:
                    logger.error(f"[RAG-File-PG] Ollama 응답 오류: {resp.status_code}")
                    raise HTTPException(
                        status_code=502, detail=f"Ollama 응답 오류: {resp.status_code}"
                    )

                data = resp.json()
                raw_content = (data.get("response") or "").strip()

                # 3-4) JSON 파싱
                try:
                    json_str = extract_json_array(raw_content)
                    batch_cases = json.loads(json_str)
                    logger.info(
                        f"[RAG-File-PG] Batch {batch_idx} 완료 - {len(batch_cases)}개 TC 생성됨"
                    )
                except Exception as e:
                    logger.error(f"[RAG-File-PG] Batch {batch_idx} JSON 파싱 실패: {e}")
                    continue

                # 3-5) ID 재부여 및 병합
                for tc in batch_cases:
                    tc["testcase_id"] = f"{id_prefix}_{global_index:03d}"
                    global_index += 1
                    all_testcases.append(tc)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[RAG-File-PG] 처리 도중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not all_testcases:
        logger.warning("[RAG-File-PG] 생성된 테스트케이스가 없음")
        raise HTTPException(
            status_code=500, detail="유효한 테스트케이스가 생성되지 않았습니다."
        )

    logger.info(
        f"[RAG-File-PG] 최종 완료 - 총 {len(all_testcases)}개 테스트케이스 반환"
    )
    return RagQAResponse(
        answer=json.dumps(all_testcases, ensure_ascii=False),
        contexts=contexts,
    )


@router.post("/generate/figma", response_model=RagQAResponse)
async def rag_generate_from_figma(
    body: RagQARequest, db: Session = Depends(get_vector_db)
) -> RagQAResponse:
    """
    Figma 화면 정보를 기반으로 QA 테스트케이스 JSON을 생성하는 엔드포인트. (Postgres 버전)
    """
    settings = get_app_settings()
    config = get_app_config(db)
    if not config:
        raise HTTPException(
            status_code=404,
            detail="DB에 시스템 설정(Configurations)이 존재하지 않습니다.",
        )

    logger.info("[RAG-Figma-PG] Figma RAG 요청 수신")

    if not figma_rag_vector_store_pg.enabled:
        logger.warning("[RAG-Figma-PG] Figma RAG 기능이 비활성화되어 있음")
        raise HTTPException(
            status_code=503, detail="Figma RAG 기능이 비활성화되어 있습니다."
        )

    # 1) 검색용 쿼리는 .env에서 가져오기
    search_query = settings.RAG_DEFAULT_QUERY
    search_top_k = settings.RAG_TOP_K

    # DB 설정 적용
    rag_batch_size = config.rag_batch_size
    n = config.rag_tc_count
    id_prefix = config.rag_tc_id_prefix
    llm_model = config.llm_model

    # DB에서 현재 소스에 해당하는 총 아이템 수 확인
    total_items = figma_rag_vector_store_pg.get_count()
    if total_items == 0:
        logger.warning("[RAG-Figma-PG] DB에 Figma 데이터가 없음")
        raise HTTPException(
            status_code=404, detail="DB에 저장된 Figma 데이터가 없습니다."
        )

    # top_k: 0 이하 → 전체 사용, 아니면 상위 N개
    if search_top_k and search_top_k > 0:
        top_k = min(search_top_k, total_items)
    else:
        top_k = total_items  # 전체 사용

    logger.info(
        "[RAG-Figma-PG] 전체 컨텍스트 수=%d, 설정 top_k=%s → 실제 사용 컨텍스트 수=%d",
        total_items,
        search_top_k,
        top_k,
    )

    # 2) RAG 벡터 검색 (PG 전용 검색 로직 사용)
    contexts = figma_rag_vector_store_pg.search(search_query, top_k=top_k)

    if not contexts:
        logger.warning(f"[RAG-Figma-PG] 검색 쿼리: {search_query} 에 대한 결과 없음")
        raise HTTPException(
            status_code=404, detail="관련 Figma 화면을 찾지 못했습니다."
        )

    # batch_size 설정
    if rag_batch_size <= 0 or rag_batch_size >= len(contexts):
        rag_batch_size = len(contexts)

    # 총 배치 수 계산
    num_batches = (len(contexts) + rag_batch_size - 1) // rag_batch_size
    logger.info(
        "[RAG-Figma-PG] 배치 실행 준비 - 전체 컨텍스트=%d, 배치 크기=%d, 총 배치 수=%d",
        len(contexts),
        rag_batch_size,
        num_batches,
    )

    # 3) 테스트케이스 개수 / ID prefix
    # n, id_prefix는 위에서 DB 설정으로 이미 가져옴
    model_name = body.model or llm_model
    logger.info(
        "[RAG-Figma-PG] 요청 n=%d, ID prefix=%s, model=%s",
        n,
        id_prefix,
        model_name,
    )

    all_testcases = []
    global_index = 1

    try:
        async with httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        ) as client:
            for batch_idx, i in enumerate(
                range(0, len(contexts), rag_batch_size), start=1
            ):
                batch = contexts[i : i + rag_batch_size]
                remaining_batches = num_batches - batch_idx
                logger.info(
                    "[RAG-Figma-PG] Batch %d/%d 시작 - 현재 배치 크기=%d, 남은 배치 수=%d",
                    batch_idx,
                    num_batches,
                    len(batch),
                    remaining_batches,
                )

                context_text_parts = []
                for idx, item in enumerate(batch, start=1):
                    meta = item.get("meta", {})
                    source = meta.get("source", "figma")
                    page_name = meta.get("page_name")
                    variant = meta.get("variant")
                    section_name = meta.get("section_name")
                    screen_path = meta.get("screen_path")

                    header = f"[{idx}] source={source}"
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

                # figma 전용 프롬프트
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
                    f"   - 'XXX 텍스트가 존재하는지 확인' 같은 정적 검증용 TC는 생성하지 마세요.\n"
                    f"2. **User Action 필수**: \n"
                    f"   - 모든 TC는 클릭, 탭, 입력 등 **사용자의 능동적 행동**을 포함해야 합니다.\n"
                    f"3. **Context 기반**: Figma 데이터에 명시된 버튼/링크/인터랙션만 검증.\n"
                    f"4. **Module 그룹화**: 화면나 섹션 단위로 'module' 필드를 통일.\n"
                    f"5. **최대 {n}개**: 의미 있는 인터랙션 테스트만 선별.\n"
                )

                full_prompt = f"{rules}### CONTEXT\n{context_text}\n### END CONTEXT\n"

                resp = await client.post(
                    "/api/generate",
                    json={"model": model_name, "prompt": full_prompt, "stream": False},
                )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_content = data.get("response", "")
                    json_str = extract_json_array(raw_content)
                    batch_cases = json.loads(json_str)

                    # 테스트케이스에 ID 할당 및 목록에 추가
                    for tc in batch_cases:
                        tc["testcase_id"] = f"{id_prefix}_{global_index:03d}"
                        global_index += 1
                        all_testcases.append(tc)

                    logger.info(
                        f"[RAG-Figma-PG] Batch {batch_idx} 완료 - {len(batch_cases)}개 TC 생성됨"
                    )

    except Exception as e:
        logger.exception("[RAG-Figma-PG] 테스트케이스 생성 중 오류 발생: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "[RAG-Figma-PG] 최종 완료 - 총 %d개 테스트케이스 반환", len(all_testcases)
    )
    return RagQAResponse(
        answer=json.dumps(all_testcases, ensure_ascii=False),
        contexts=contexts,
    )


@router.post("/generate/all", response_model=RagQAResponse)
async def rag_generate_all(
    body: RagQARequest, db: Session = Depends(get_vector_db)
) -> RagQAResponse:
    """
    [통합 RAG] 파일과 피그마 데이터를 모두 통합하여 검색하고,
    QA 테스트케이스를 생성하며 실시간 진행상황을 DB(History)에 기록합니다.
    """
    settings = get_app_settings()
    config = get_app_config(db)
    if not config:
        raise HTTPException(
            status_code=404,
            detail="DB에 시스템 설정(Configurations)이 존재하지 않습니다.",
        )

    # 1) 공통 설정 로드
    search_query = settings.RAG_DEFAULT_QUERY
    search_top_k = settings.RAG_TOP_K
    rag_batch_size = config.rag_batch_size
    n = config.rag_tc_count
    id_prefix = body.tcPrefix or config.rag_tc_id_prefix
    llm_model = body.model or config.llm_model or "llama3.1:8b"
    language = body.language or "한글"

    # DB에서 전체 소스(file + figma)에 해당하는 총 아이템 수 확인
    total_items = db.query(RagEmbedding).count()
    if total_items == 0:
        logger.warning("[RAG-All-PG] DB에 데이터가 없음")
        raise HTTPException(status_code=404, detail="DB에 저장된 데이터가 없습니다.")

    # top_k: 0 이하 → 전체 사용, 아니면 상위 N개
    if search_top_k and search_top_k > 0:
        top_k = min(search_top_k, total_items)
    else:
        top_k = total_items  # 전체 사용

    logger.info(
        "[RAG-All-PG] 전체 컨텍스트 수=%d, 설정 top_k=%s → 실제 사용 top_k=%d",
        total_items,
        search_top_k,
        top_k,
    )

    # 2) 통합 벡터 검색 (File + Figma)
    query_vec = embedding_service.embed_query(search_query)[0].tolist()
    results = search_all_sources(db, query_vec, top_k=top_k)

    if not results:
        raise HTTPException(status_code=404, detail="관련 내용을 찾지 못했습니다.")

    contexts = [{"text": r.text, "meta": r.metadata_json} for r in results]
    num_batches = (len(contexts) + rag_batch_size - 1) // rag_batch_size

    # 3) 실행 이력 DB 생성 (Status: Running)
    db_history = create_history(
        db,
        title=f"{body.title}",
        source_type="all",
        total_batches=num_batches,
        model_name=llm_model,
    )

    all_testcases = []
    global_index = 1

    # Register current task for cancellation
    current_task = asyncio.current_task()
    _active_tasks[db_history.id] = current_task

    try:
        async with httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        ) as client:
            for batch_idx, i in enumerate(
                range(0, len(contexts), rag_batch_size), start=1
            ):
                batch = contexts[i : i + rag_batch_size]

                # 실시간 프로그레스 업데이트
                update_progress(
                    db,
                    db_history.id,
                    batch_idx,
                    log_msg=f"Batch {batch_idx}/{num_batches} 생성 중...",
                )

                # 컨텍스트 구성
                context_text = "\n\n".join(
                    [f"[{idx + 1}] {c['text']}" for idx, c in enumerate(batch)]
                )

                # [통합용 프롬프트] 파일의 명세 정보와 피그마의 UI 정보를 모두 고려하도록 설계
                prompt = (
                    f"역할: 숙련된 QA 엔지니어\n"
                    f"작업: [CONTEXT](문서+Figma)를 분석해 '사용자 흐름' 중심 테스트케이스 작성\n"
                    f"출력: 오직 JSON 배열만 반환 (설명/마크다운 제거)\n\n"
                    f"규칙:\n"
                    f"1. 언어: 반드시 {language}로 작성\n"
                    f"2. ID: '{id_prefix}_###' 형식 (001부터 순차 증가)\n"
                    f"3. Steps: [행동] 클릭/입력/선택 및 [확인] UI 변화를 구체적으로 기술\n"
                    f"4. 정합성: 문서 로직과 Figma UI 요소를 매핑. 추측 금지\n"
                    f"5. 수량: 유의미한 케이스로 최대 {n}개 (중복 없이 {n}개에 가깝게 생성)\n\n"
                    f"스키마:\n"
                    f'[{{"testcase_id":"ID","module":"화면명","title":"제목","preconditions":"사전조건","steps":["1...","2..."],"expected_result":"결과","priority":"High|Medium|Low"}}]\n\n'
                    f"CONTEXT:\n{context_text}"
                )

                # LLM 호출
                resp = await client.post(
                    "/api/generate",
                    json={"model": llm_model, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_content = (data.get("response") or "").strip()

                    try:
                        # Extract JSON array from the response
                        json_str = extract_json_array(raw_content)
                        batch_cases = (
                            json.loads(json_str)
                            if isinstance(json_str, str)
                            else json_str
                        )

                        if not isinstance(batch_cases, list):
                            batch_cases = [batch_cases]

                        for tc in batch_cases:
                            if not isinstance(tc, dict):
                                logger.warning(
                                    "Skipping non-dict test case: %s", str(tc)[:200]
                                )
                                continue

                            tc["testcase_id"] = f"{id_prefix}_{global_index:03d}"
                            global_index += 1
                            all_testcases.append(tc)

                    except Exception as e:
                        logger.exception("Failed to parse test cases: %s", str(e))
                        raise

        # 4) 성공 처리
        complete_history(
            db,
            db_history.id,
            status="success",
            summary=f"통합 테스트케이스 {len(all_testcases)}개 생성 완료",
            result_data=all_testcases,
        )

    except asyncio.CancelledError:
        logger.info(f"[RAG-All-PG] Task {db_history.id} cancelled by user")
        # History status update is handled in cancel_generation endpoint
        raise
    except Exception as e:
        logger.exception("통합 RAG 생성 실패")
        complete_history(db, db_history.id, status="failed", summary=f"오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove from active tasks
        _active_tasks.pop(db_history.id, None)

    return RagQAResponse(
        answer=json.dumps(all_testcases, ensure_ascii=False),
        contexts=contexts,
    )
