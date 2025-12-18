from fastapi import APIRouter, Depends
from app.api.deps import get_app_settings
from app.services.ollama_client import OllamaClient
from app.services.mcp_client import _chat_with_mcp_and_ollama
from app.schemas.ollama import OllamaChatRequest, OllamaChatResponse

import httpx

router = APIRouter(prefix="/mcp", tags=["mcp"])
Settings = get_app_settings()


@router.post("/chat", response_model=OllamaChatResponse)
async def chat_with_mcp_ollama(
    body: OllamaChatRequest,
    settings: Settings = Depends(get_app_settings),
):
    """
    기존: Ollama로 바로 프록시
    지금: 필요 시 MCP 툴까지 사용하는 오케스트레이션
    """
    client = OllamaClient(settings)

    # 가장 단순하게: 마지막 메시지 content를 user_input으로 사용
    if body.messages:
        last_msg = body.messages[-1]
        user_input = last_msg.content
    else:
        user_input = ""

    # OllamaChatMessage 객체를 딕셔너리로 변환
    messages_dict = [msg.model_dump() for msg in body.messages]

    if not user_input:
        return OllamaChatResponse(
            success=False,
            output=None,
            raw={"error": "마지막 메시지 content가 비어 있습니다."},
        )

    try:
        result = await _chat_with_mcp_and_ollama(
            ollama_client=client,
            user_input=user_input,
            model=body.model,
            messages_dict=messages_dict,
            options=body.options,
        )

        final_text = result.get("final_text")
        if not final_text and "first_llm_response" in result:
            first = result["first_llm_response"]
            final_text = first.get("message", {}).get("content")

        return OllamaChatResponse(
            success=True,
            output=final_text,
            raw=result,
        )

    except httpx.HTTPError as e:
        return OllamaChatResponse(
            success=False,
            output=None,
            raw={"error": f"Ollama 통신 에러: {str(e)}"},
        )
    except Exception as e:
        return OllamaChatResponse(
            success=False,
            output=None,
            raw={"error": f"서버 내부 에러: {repr(e)}"},
        )
