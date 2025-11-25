import asyncio
import json
from typing import Optional, Dict, Any, List
import httpx
import sys
from pathlib import Path
import re

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

# ⚠️ 실제 Ollama 엔드포인트가 /api/chat 이면 이렇게 쓰는 게 더 안전해:
# OLLAMA_URL = "http://59.29.234.26:11434/api/chat"
OLLAMA_URL = "http://59.29.234.26:11434/api/chat"
OLLAMA_MODEL = "exaone3.5:2.4b"


class OllamaMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_mcp_server(self, command: str, args: List[str]):
        """
        MCP 서버(네 weather / user-server 등)를 stdio로 실행 & 연결
        예: command=sys.executable, args=["/abs/path/to/server.py", "mcp"]
        """
        params = StdioServerParameters(command=command, args=args)

        # stdio_client는 async context manager이므로 이렇게 써야 함
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(params)
        )
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        print("MCP 서버 연결 완료!")

        tools_response = await self.session.list_tools()
        print("🔧 MCP 서버 툴 목록:", [t.name for t in tools_response.tools])
        return tools_response.tools

    async def call_ollama(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ollama /api/chat 호출
        messages 형식은 Ollama 스타일로 맞춘다.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def chat_once(self, user_input: str):
        """
        한 번의 질문에 대해:
        1) Ollama에게 "툴 설명 + 규칙"을 주고
        2) 필요하면 MCP 툴 호출해서
        3) 최종 답변까지 받아오기
        """

        if self.session is None:
            raise RuntimeError("MCP 세션이 아직 초기화되지 않았어요.")

        # 1. MCP 툴 목록 불러오기
        tools_response = await self.session.list_tools()
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

        # 2. Ollama에게 규칙 설명
        system_prompt = (
            "너는 MCP 툴을 사용할 수 있는 비서야.\n"
            "지금부터 사용할 수 있는 툴 목록과 입력 스키마를 줄게.\n"
            "툴을 써야 한다고 판단되면, 다음과 같은 **순수 JSON만** 출력해:\n\n"
            '{\n'
            '  "tool_name": "<툴 이름>",\n'
            '  "arguments": { ... }\n'
            '}\n\n'
            "툴이 필요 없으면 그냥 자연어로 대답해.\n"
        )

        tool_info_str = json.dumps(tools_desc, ensure_ascii=False, indent=2)

        messages = [
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

        ollama_resp = await self.call_ollama(messages)
        assistant_msg = ollama_resp["message"]["content"]

        # Ollama 응답이 JSON(툴 호출)인지, 그냥 답변인지 판별
        tool_call = self._try_parse_tool_call(assistant_msg)

        if tool_call is None:
            # 그냥 답변
            print("\n🤖 LLM 답변:")
            print(assistant_msg)
            return

        # 3. 툴 호출
        tool_name = tool_call["tool_name"]
        arguments = tool_call.get("arguments", {})

        print(f"\n🔧 툴 호출 요청: {tool_name}({arguments})")

        tool_result = await self.session.call_tool(tool_name, arguments)

        # 4. 툴 결과를 가지고 Ollama에게 최종 답변 요청
        tool_result_str = json.dumps(
            tool_result.content, ensure_ascii=False, indent=2
        )

        messages2 = [
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

        final_resp = await self.call_ollama(messages2)
        final_text = final_resp["message"]["content"]

        print("\n✅ 최종 답변:")
        print(final_text)

    def _try_parse_tool_call(
        self, text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Ollama 응답이 JSON형식의 툴 호출인지 시도해서 파싱.
        제대로 된 JSON 아니면 None 리턴.
        """
        # 먼저 문자열로 정규화
        if not isinstance(text, str):
            text = str(text)

        raw = text.strip()
        
        if raw.startswith("```"):
            m = re.search(
                r"```(?:json)?\s*(\{.*\})\s*```",
                raw,
                re.DOTALL,
            )
            if m:
                raw = m.group(1).strip()

        # 2) 여전히 { 로 시작하는 문자열만 JSON으로 간주
        if not raw.startswith("{"):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if "tool_name" in data and "arguments" in data:
            return data
        return None


async def main():
    client = OllamaMCPClient()

    # ✅ 여기서 더 이상 "python" 같은 하드코딩 X
    #    현재 실행 중인 파이썬 인터프리터 경로 그대로 사용
    command = sys.executable

    # ✅ MCP 서버 스크립트 경로 (반드시 **실제 MCP 서버 파일**로 바꿔줘야 함!)
    server_script = Path(__file__).with_name("mcp_server.py")
    args = [str(server_script)]  # 필요하면 [".../server.py", "mcp"] 처럼 인자 추가

    await client.connect_to_mcp_server(command, args)

    while True:
        q = input("\n💬 질문 (quit 입력 시 종료): ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        await client.chat_once(q)


if __name__ == "__main__":
    asyncio.run(main())
