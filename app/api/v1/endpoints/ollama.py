# FastAPI 라우터 임포트
from fastapi import APIRouter, Depends, HTTPException

# 상태 코드 상수 임포트
from starlette import status

# 설정 의존성 임포트
from app.api.deps import get_app_settings
from sqlalchemy.orm import Session
from app.db.database import get_vector_db
from app.crud.config import get_app_config

# 설정 타입 임포트
from app.core.config import Settings

# 클라이언트 임포트
from app.services.ollama_client import OllamaClient

# 스키마 임포트
from app.schemas.ollama import (
    OllamaStatus,
    OllamaModels,
    OllamaModelItem,
    OllamaChatRequest,
    OllamaChatResponse,
)

# 예외 타입 임포트
import httpx
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ollama", tags=["ollama"])


# 상태 조회 엔드포인트
@router.get("/status", response_model=OllamaStatus)
async def get_ollama_status(settings: Settings = Depends(get_app_settings)):
    # 클라이언트 생성
    client = OllamaClient(settings)
    try:
        # 상태 조회 실행
        data = await client.get_status()
        # 결과 반환
        return OllamaStatus(**data)
    # 통신 예외 처리
    except httpx.HTTPError as e:
        # 접근 불가 응답 구성
        return OllamaStatus(reachable=False, version=None, raw={"error": str(e)})


# 모델 리스트 엔드포인트
@router.get("/models", response_model=OllamaModels)
async def get_ollama_models(settings: Settings = Depends(get_app_settings)):
    # 클라이언트 생성
    client = OllamaClient(settings)
    try:
        # 모델 리스트 조회
        data = await client.list_models()
        # Ollama는 {"models":[{name:...,digest:...,size:...},...]} 형태를 반환
        models_raw = data.get("models", [])
        # 스키마로 변환
        models = [OllamaModelItem(**m) for m in models_raw]
        # 결과 래핑
        return OllamaModels(models=models)
    # 통신 예외 처리
    except httpx.HTTPError as e:
        # 502로 위임
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama upstream error: {str(e)}",
        )


# 모델 질문 전송 엔드포인트
@router.post("/chat", response_model=OllamaChatResponse)
async def chat_with_ollama(
    body: OllamaChatRequest,
    settings: Settings = Depends(get_app_settings),
    db: Session = Depends(get_vector_db),
):
    # 클라이언트 생성
    client = OllamaClient(settings)

    # DB에서 최신 설정 가져오기
    config = get_app_config(db)
    # 바디에 모델이 있으면 사용하고, 없으면 DB 설정을 우선 사용 (DB 설정도 없으면 바디의 기본값 등 활용)
    target_model = body.model
    if config and config.llm_model:
        target_model = config.llm_model
        logger.info(f"[Ollama-Chat] DB 설정 모델 사용: {target_model}")
    else:
        logger.info(f"[Ollama-Chat] 요청 바디 모델 사용: {target_model}")

    try:
        # chat 호출
        # OllamaChatMessage 객체를 딕셔너리로 변환
        messages_dict = [msg.model_dump() for msg in body.messages]
        data = await client.chat_with_messages(
            messages=messages_dict,
            model=target_model,
            stream=body.stream,
            options=body.options,
        )
        output = data.get("messages", {}).get("content")
        return OllamaChatResponse(
            success=True,
            output=output,
            raw=data,
        )

    # 통신 예외 처리
    except httpx.HTTPError as e:
        # 실패 응답 구성
        return OllamaChatResponse(
            success=False,
            output=None,
            raw={"error": str(e)},
        )
