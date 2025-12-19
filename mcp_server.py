#!/usr/bin/env python3
"""
MCP 서버 - tbell.ai_model / eval_session_plan 테이블에서 실제 데이터를 조회하는 서버
"""

import asyncio
import json
from typing import Any, Dict, Optional, List

from mcp.server import Server, InitializationOptions, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Tool,
    TextContent,
)
from app.core.logging import get_logger
from app.crud import mcp as mcp_crud

# MCP 서버 인스턴스 생성
server = Server("ai-model-mcp-server")
logger = get_logger(__name__)


# ===== 비동기 래퍼 함수들 =====
async def fetch_all_models(only_used: bool = False) -> List[Dict[str, Any]]:
    """ai_model 테이블에서 전체 모델 목록을 비동기로 조회합니다."""
    return await asyncio.to_thread(mcp_crud.fetch_all_models_sync, only_used)


async def fetch_model_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """지정된 이름의 AI 모델을 비동기로 조회합니다."""
    return await asyncio.to_thread(mcp_crud.fetch_model_by_name_sync, model_name)


async def fetch_eval_session_stats() -> Dict[str, Any]:
    """평가 세션의 상태별 통계를 비동기로 조회합니다."""
    return await asyncio.to_thread(mcp_crud.fetch_eval_session_stats_sync)


# ===== MCP 도구 정의 =====
@server.list_tools()
async def list_tools() -> ListToolsResult:
    """사용 가능한 MCP 도구 목록을 반환합니다."""
    tools = [
        # 1. AI 모델 목록 조회 도구
        Tool(
            name="list_ai_models",
            description="ai_model 테이블의 모델 전체 목록을 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "only_used": {
                        "type": "boolean",
                        "description": "true이면 is_used = true 인 모델만 조회합니다.",
                    }
                },
                "required": [],
            },
        ),
        # 2. 특정 AI 모델 조회 도구
        Tool(
            name="get_ai_model_by_name",
            description="model_name으로 특정 AI 모델 한 개를 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "조회할 모델 이름 (ai_model.model_name)",
                    }
                },
                "required": ["model_name"],
            },
        ),
        # 3. 평가 세션 통계 조회 도구
        Tool(
            name="get_eval_session_stats",
            description=(
                "eval_session_plan 테이블에서 "
                "RUNNING : '수행중' / DONE : '수행 완료' / ERROR : '수행 중 오류' 상태별 평가 세션 개수를 집계합니다."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]
    return ListToolsResult(tools=tools)


# ===== MCP 도구 호출 처리 =====
@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """MCP 도구 호출을 처리하고 결과를 반환합니다."""
    try:
        # 1) AI 모델 전체 목록 조회
        if name == "list_ai_models":
            only_used = bool(arguments.get("only_used", False))
            models = await fetch_all_models(only_used=only_used)
            text = json.dumps(models, ensure_ascii=False, indent=2)
            return CallToolResult(
                content=[TextContent(type="text", text=f"ai_model 목록:\n{text}")]
            )

        # 2) 모델 이름으로 단일 모델 조회
        elif name == "get_ai_model_by_name":
            model_name = arguments.get("model_name")
            if not model_name:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text="오류: model_name 인자가 필요합니다."
                        )
                    ],
                    isError=True,
                )

            model = await fetch_model_by_name(str(model_name))
            if model is None:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"오류: model_name='{model_name}' 인 AI 모델을 찾을 수 없습니다.",
                        )
                    ],
                    isError=True,
                )

            text = json.dumps(model, ensure_ascii=False, indent=2)
            return CallToolResult(
                content=[TextContent(type="text", text=f"AI 모델 정보:\n{text}")]
            )

        # 3) 평가 세션 통계 조회
        elif name == "get_eval_session_stats":
            stats = await fetch_eval_session_stats()
            text = json.dumps(stats, ensure_ascii=False, indent=2)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"eval_session_plan 상태 집계:\n{text}"
                    )
                ]
            )

        # 등록되지 않은 도구 요청 시
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=f"오류: '{name}' 도구를 찾을 수 없습니다."
                )
            ],
            isError=True,
        )

    except Exception as e:
        logger.exception("도구 실행 중 오류 발생")
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"도구 실행 중 오류 발생: {str(e)}")
            ],
            isError=True,
        )


async def main():
    """MCP 서버를 시작하는 메인 함수"""
    logger.info("🚀 MCP 서버(ai_model + eval_session_plan) 시작 중...")

    async with stdio_server() as (read_stream, write_stream):
        init_options = InitializationOptions(
            server_name="ai-model-mcp-server",
            server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        )

        await server.run(read_stream, write_stream, init_options)

    logger.info("🛑 MCP 서버 종료")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ MCP 서버 실행 중 오류 발생: {e}", exc_info=True)
