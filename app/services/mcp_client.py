import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.types import TextContent, CallToolResult
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

from app.utils.mcp_parser import _parse_tool_call
from app.services.ollama_client import OllamaClient
from app.core.config import Settings
from app.core.logging import get_logger

# 올바른 방법:
settings = Settings()
MCP_SERVER_PATH = Path(__file__).resolve().parents[2] / settings.MCP_SERVER
MCP_SERVER_COMMAND = settings.MCP_SERVER_COMMAND
MCP_SERVER_ARGS = [str(MCP_SERVER_PATH)]

logger = get_logger(__name__)


class MCPClientManager:
    def __init__(self) -> None:
        self._session: Optional[ClientSession] = None
        self._stdio = None
        self._write = None
        self._exit_stack = AsyncExitStack()
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session

        async with self._lock:
            if self._session is not None:
                return self._session

            params = StdioServerParameters(
                command=MCP_SERVER_COMMAND,
                args=MCP_SERVER_ARGS,
                env=None,
            )

            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            self._stdio, self._write = read_stream, write_stream

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(self._stdio, self._write)
            )

            try:
                await self._session.initialize()
                print(f"[MCP] 서버 연결 완료: {MCP_SERVER_COMMAND} {MCP_SERVER_ARGS}")
            except Exception as e:
                logger.error(f"❌ MCP 연결 실패: {e}", exc_info=True)
                raise

            return self._session

    async def list_tools(self):
        logger.info("📋 MCP 도구 목록 조회 중...")
        session = await self._ensure_session()
        tools = await session.list_tools()
        logger.info(f"✅ 도구 목록 획득: {len(tools.tools)}개")
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        session = await self._ensure_session()
        return await session.call_tool(name, arguments)


MCP_MANAGER = MCPClientManager()


# TextContent → JSON-serializable dict 로 평탄화
def _flatten_tool_result(tool_result: CallToolResult) -> Dict[str, Any]:
    flat_contents = []

    for c in tool_result.content:
        if isinstance(c, TextContent):
            flat_contents.append(
                {
                    "type": "text",
                    "text": c.text,
                }
            )
        else:
            if hasattr(c, "model_dump"):
                flat_contents.append(c.model_dump())
            else:
                flat_contents.append({"repr": repr(c)})

    return {
        "isError": getattr(tool_result, "isError", False),
        "content": flat_contents,
    }


async def _chat_with_mcp_and_ollama(
    ollama_client: OllamaClient,
    user_input: str,
    model: str,
    messages_dict: Optional[list[Dict[str, str]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    options = options or {}
    messages_dict = messages_dict or []

    # 1. MCP 툴 목록
    tools_response = await MCP_MANAGER.list_tools()
    tools = tools_response.tools

    tools_desc = []
    for t in tools:
        tools_desc.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
        )

    # 2. 첫 번째 LLM 호출: 툴 필요 여부 & 인자 추론
    system_prompt = (
        "너는 MCP 기능을 사용할 수 있는 비서야.\n"
        "항상 두 가지 모드 중 하나로만 답해야 한다.\n\n"
        "1) JSON 모드 (도구 사용)\n"
        "- 사용 가능한 기능 목록과 스키마를 보고, 특정 기능을 호출해야 한다고 판단되면\n"
        "  아래 형식의 '순수 JSON'만 출력해라.\n"
        "{\n"
        '  "tool_name": "<툴 이름>",\n'
        '  "arguments": { ... }\n'
        "}\n"
        "- JSON 바깥에 자연어, 설명, 코드블록, 주석은 한 글자도 섞지 마.\n\n"
        "2) 자연어 모드 (도구 미사용 또는 지원 불가)\n"
        "- 도구가 필요 없다고 판단되거나, 사용 가능한 기능만으로는 사용자의 요청을 만족시킬 수 없으면\n"
        "  한국어 한 문단으로만 짧게 답해라.\n"
        "- 이때는 '툴', 'tool', '도구', 'MCP', '엔드포인트', '스키마', 'JSON',\n"
        "  '호출 결과', '요청/응답', '시스템' 같은 단어를 절대 쓰지 마.\n"
        "- '도구를 쓰지 않고 답변하자면', '현재 시스템에서', '직접적인 도구 활용 없이도'와 같은\n"
        "  메타 설명도 절대 쓰지 마.\n"
        "- 최대 2~3문장 안에서, 사용자가 궁금해하는 내용만 자연스럽게 말해라.\n"
        "- '다음과 같습니다', '아래와 같이' 같은 말도 쓰지 말고, 바로 내용 문장으로 시작해라.\n\n"
        "추가 규칙:\n"
        "- 사용자의 요청을 현재 기능 목록으로는 처리할 수 없으면,\n"
        "  짧게 '그 정보는 여기서는 알 수 없다'는 취지로 말하고,\n"
        "  한 문장 정도로 너가 도와줄 수 있는 범위를 자연스럽게 덧붙여라.\n"
        "- 이때도 툴 이름이나 기술적인 용어는 쓰지 말고,\n"
        "  사람이 이해하기 쉬운 표현으로만 정리해라.\n"
    )

    tool_info_str = json.dumps(tools_desc, ensure_ascii=False, indent=2)

    first_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "사용 가능한 기능 목록과 입력 스키마:\n"
                f"{tool_info_str}\n\n"
                f"사용자 질문: {user_input}\n\n"
                "위 정보를 참고해서, 도구를 사용해야 한다고 판단되면 JSON 모드로,\n"
                "도구 없이 답할 수 있거나, 지금 기능으로는 요청을 처리할 수 없다고 판단되면\n"
                "담백하게 거짓말 하지말고 그냥 제공할 수 없는 정보라고 짧게 한글로 답해줘."
            ),
        },
    ]

    first_resp = await ollama_client.chat_with_messages(
        messages=first_messages,
        model=model,
        stream=False,
        options=options,
    )

    assistant_msg = first_resp.get("message", {}).get("content", "")
    tool_call = _parse_tool_call(assistant_msg)

    # 2-1. 툴 호출이 아니면 → 첫 답변 그대로 사용
    if tool_call is None:
        return {
            "mode": "no_tool",
            "first_llm_response": first_resp,
            "final_text": assistant_msg,
        }

    # 3. MCP tool 호출
    tool_name = tool_call["tool_name"]
    arguments = tool_call.get("arguments", {})

    try:
        tool_result = await MCP_MANAGER.call_tool(tool_name, arguments)
    except McpError as e:
        return {
            "mode": "tool_error",
            "first_llm_response": first_resp,
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "error": str(e),
        }

    # TextContent 등을 JSON-serializable dict 로 변환
    flat_tool_result = _flatten_tool_result(tool_result)

    # LLM에 넣을 문자열은 content 부분만 보기 좋게
    tool_result_str = json.dumps(
        flat_tool_result["content"], ensure_ascii=False, indent=2
    )

    # 4. 툴 결과를 가지고 두 번째 LLM 호출
    second_messages = [
        {
            "role": "system",
            "content": (
                "너는 숨겨진 내부 값을 참고해서 한국어로 짧은 답변만 만들어주는 비서야.\n\n"
                "규칙:\n"
                "- 답변에서는 다음 단어/표현을 절대 쓰지 마: "
                "'MCP', '툴', 'tool', '도구', 'API', '엔드포인트', "
                "'호출 결과', '쿼리 결과', '내부 시스템', '시스템에서 조회한', "
                "'데이터베이스', 'DB', '데이터', '비공개', 'JSON', '로그', '통계'.\n"
                "- 출처나 처리 과정, 함수/쿼리 이름, 어떤 값이 어디서 왔는지 등의 설명을 하지 마.\n"
                "- 마치 네가 원래 알고 있던 사실을 말하듯 자연스럽게, 사용자의 질문에만 답해.\n"
                "- 기본 답변 길이는 최대 2~3문장, 한 문단(줄바꿈 없이)으로만 답해.\n"
                "- '다음과 같습니다', '아래와 같이', '제공된 정보에 따르면' 같은 메타 표현 대신 바로 내용으로 시작해."
            ),
        },
        {
            "role": "user",
            "content": (
                "다음은 사용자가 실제로 했던 질문이야.\n"
                f"{user_input}\n\n"
                "그리고 아래에는 너만 참고하는 추가 정보가 있다.\n"
                "이 추가 정보가 있다는 사실이나 출처를 답변에서 절대 언급하지 마.\n\n"
                "❗ 매우 중요:\n"
                "- 위에서 언급한 금지어들은 답변에 절대 포함하지 마.\n"
                "- '정보', '데이터', '자료', '시스템', '통계', '결과' 등 "
                "추가 정보를 참고했다는 느낌을 주는 표현도 최대한 쓰지 마.\n"
                "- 사용자가 궁금해하는 내용만 2~3문장으로 간단하게, 자연스럽게 설명해.\n"
                "- 질문을 다시 반복하지 말고, 단순한 결론/설명만 말해.\n\n"
                "참고용 값:\n"
                f"{tool_result_str}\n\n"
                "위 참고용 값을 바탕으로, 사용자의 질문에 대해 한국어로만 짧게 답해줘.\n"
                "목록/필드명을 기계적으로 나열하지 말고, 사람이 읽기 편한 문장으로 요약해줘."
            ),
        },
    ]

    second_resp = await ollama_client.chat_with_messages(
        messages=second_messages,
        model=model,
        stream=False,
        options=options,
    )

    final_text = second_resp.get("message", {}).get("content", "")

    return {
        "mode": "tool_used",
        "first_llm_response": first_resp,
        "tool_name": tool_name,
        "tool_arguments": arguments,
        "tool_result": flat_tool_result,
        "second_llm_response": second_resp,
        "final_text": final_text,
    }
