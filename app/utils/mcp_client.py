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
from app.utils.ollama_client import OllamaClient
from app.core.config import Settings
import logging
from app.core.logging import get_logger

# ✅ 올바른 방법:
settings = Settings()
MCP_SERVER_PATH = Path(__file__).resolve().parents[2] / settings.MCP_SERVER
MCP_SERVER_COMMAND = settings.MCP_SERVER_COMMAND
MCP_SERVER_ARGS = [str(MCP_SERVER_PATH)]

print(f"[MCP Client] 서버 경로: {MCP_SERVER_PATH}")
print(f"[MCP Client] 실행 명령어: {MCP_SERVER_COMMAND}")
print(f"[MCP Client] 파일 존재 여부: {MCP_SERVER_PATH.exists()}")


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

            print("[MCP] 세션 초기화 시작...")
            try:
                await self._session.initialize()
                print(f"[MCP] 서버 연결 완료: {MCP_SERVER_COMMAND} {MCP_SERVER_ARGS}")
            except Exception as e:
                logger.error(f"❌ MCP 연결 실패: {e}", exc_info=True)
                raise

            return self._session

    async def list_tools(self):
        logger.debug("📋 MCP 도구 목록 조회 중...")
        session = await self._ensure_session()
        tools = await session.list_tools()
        logger.info(f"✅ 도구 목록 획득: {len(tools.tools)}개")
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        session = await self._ensure_session()
        return await session.call_tool(name, arguments)


MCP_MANAGER = MCPClientManager()


# ✅ TextContent → JSON-serializable dict 로 평탄화
def _flatten_tool_result(tool_result: CallToolResult) -> Dict[str, Any]:
    flat_contents = []

    for c in tool_result.content:
        if isinstance(c, TextContent):
            flat_contents.append({
                "type": "text",
                "text": c.text,
            })
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
        "너는 MCP 툴을 사용할 수 있는 비서야.\n"
        "지금부터 사용할 수 있는 툴 목록과 입력 스키마를 줄게.\n"
        "툴을 써야 한다고 판단되면, 다음과 같은 **순수 JSON만** 출력해:\n\n"
        "{\n"
        '  "tool_name": "<툴 이름>",\n'
        '  "arguments": { ... }\n'
        "}\n\n"
        "툴이 필요 없으면 그냥 자연어로 대답해.\n"
    )

    tool_info_str = json.dumps(tools_desc, ensure_ascii=False, indent=2)

    first_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "사용 가능한 MCP 툴 목록과 스키마:\n"
                f"{tool_info_str}\n\n"
                f"사용자 질문: {user_input}\n\n"
                "툴이 필요하면 위 규칙에 맞춰 JSON만 출력하고,\n"
                "필요 없으면 자연어로 바로 답해줘."
            ),
        },
    ]
    print("[First LLM Request] LLM에게 tools 정보 전달")
    print(first_messages)

    first_resp = await ollama_client.chat_with_messages(
        messages=first_messages,
        model=model,
        stream=False,
        options=options,
    )
    print("[First LLM Response] LLM이 tool 선택")
    print(first_resp)

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

    # ✅ TextContent 등을 JSON-serializable dict 로 변환
    flat_tool_result = _flatten_tool_result(tool_result)

    # LLM에 넣을 문자열은 content 부분만 보기 좋게
    tool_result_str = json.dumps(
        flat_tool_result["content"], ensure_ascii=False, indent=2
    )

    # 4. 툴 결과를 가지고 두 번째 LLM 호출
    second_messages = [
        {"role": "system", "content": "너는 MCP 툴 결과를 해석해주는 비서야."},
        {
            "role": "user",
            "content": (
                f"원래 사용자 질문: {user_input}\n\n"
                f"아래는 MCP 툴 '{tool_name}' 호출 결과야:\n"
                f"{tool_result_str}\n\n"
                "사용자에게 이해하기 쉽게 한국어로 답변을 정리해줘."
            ),
        },
    ]

    print("[Second LLM Request] LLM에게 tool 결과 전달")
    print(second_messages)

    second_resp = await ollama_client.chat_with_messages(
        messages=second_messages,
        model=model,
        stream=False,
        options=options,
    )

    print("[Second LLM Response] LLM이 tool 결과 해석 최종 답변")
    print(second_resp)

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
