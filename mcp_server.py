#!/usr/bin/env python3
"""
MCP 서버 - tbell.ai_model / eval_session_plan 테이블에서 실제 데이터를 조회하는 서버 (pg8000 사용)
"""

import asyncio
import json
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse

import pg8000
from mcp.server import Server, InitializationOptions, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Tool,
    TextContent,
)
from app.core.logging import get_logger
from app.core.config import get_settings

settings = get_settings()
server = Server("ai-model-mcp-server")
logger = get_logger(__name__)

# ===== DB 설정 =====
DB_DSN = settings.MCP_DB_CON


def parse_pg_dsn(dsn: str) -> Dict[str, Any]:
    """postgresql://user:pass@host:port/db 형태 DSN을 pg8000 connect 인자로 변환"""
    r = urlparse(dsn)
    return {
        "user": r.username,
        "password": r.password,
        "host": r.hostname or "localhost",
        "port": r.port or 5432,
        "database": (r.path or "").lstrip("/") or None,
    }


# ===== 동기 쿼리를 async에서 쓰기 위해 to_thread로 감싸는 동기 함수들 =====
def _fetch_all_models_sync(only_used: bool = False) -> List[Dict[str, Any]]:
    """ai_model 테이블 전체 목록(옵션: only_used) 동기 조회"""
    cfg = parse_pg_dsn(DB_DSN)
    conn = pg8000.connect(**cfg)
    try:
        cur = conn.cursor()
        query = """
            SELECT
                id,
                model_name,
                model_version,
                is_used,
                created_at,
                updated_at
            FROM public.ai_model
        """
        if only_used:
            query += " WHERE is_used = TRUE"
        query += " ORDER BY id ASC"

        cur.execute(query)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

        result: List[Dict[str, Any]] = []
        for row in rows:
            r = dict(zip(cols, row))
            result.append(
                {
                    "id": r["id"],
                    "model_name": r["model_name"],
                    "model_version": r.get("model_version"),
                    "is_used": r["is_used"],
                    "created_at": r["created_at"].isoformat()
                    if r.get("created_at")
                    else None,
                    "updated_at": r["updated_at"].isoformat()
                    if r.get("updated_at")
                    else None,
                }
            )
        return result
    finally:
        conn.close()


def _fetch_model_by_name_sync(
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """model_name으로 ai_model 단건 동기 조회"""
    cfg = parse_pg_dsn(DB_DSN)
    conn = pg8000.connect(**cfg)
    try:
        cur = conn.cursor()
        query = """
            SELECT
                id,
                model_name,
                model_version,
                is_used,
                created_at,
                updated_at
            FROM public.ai_model
            WHERE model_name = %s
        """
        cur.execute(query, (model_name,))
        row = cur.fetchone()
        if row is None:
            return None

        cols = [d[0] for d in cur.description]
        r = dict(zip(cols, row))
        return {
            "id": r["id"],
            "model_name": r["model_name"],
            "model_version": r.get("model_version"),
            "is_used": r["is_used"],
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
        }
    finally:
        conn.close()


def _fetch_eval_session_stats_sync() -> Dict[str, Any]:
    """
    eval_session_plan에서 상태별 개수 집계 동기 조회.

    RUNNING / DONE / ERROR 기준으로 개수를 리턴하고,
    혹시 다른 상태가 있으면 raw 항목에 그대로 포함.
    """
    cfg = parse_pg_dsn(DB_DSN)
    conn = pg8000.connect(**cfg)
    try:
        cur = conn.cursor()
        query = """
            SELECT eval_status, COUNT(*) AS cnt
            FROM public.eval_session_plan
            GROUP BY eval_status
        """
        cur.execute(query)
        rows = cur.fetchall()

        # 기본값 0으로 세팅
        counts = {
            "RUNNING": 0,
            "DONE": 0,
            "ERROR": 0,
        }
        raw: List[Dict[str, Any]] = []

        for status, cnt in rows:
            status_str = (status or "").upper()
            raw.append(
                {
                    "eval_status": status,
                    "count": int(cnt),
                }
            )
            if status_str in counts:
                counts[status_str] = int(cnt)

        return {
            "running": counts["RUNNING"],
            "done": counts["DONE"],
            "error": counts["ERROR"],
            "raw": raw,
        }
    finally:
        conn.close()


# ===== async wrapper =====
async def fetch_all_models(only_used: bool = False) -> List[Dict[str, Any]]:
    """ai_model 전체 목록 async 래핑"""
    return await asyncio.to_thread(_fetch_all_models_sync, only_used)


async def fetch_model_by_name(
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """model_name으로 ai_model 단건 async 래핑"""
    return await asyncio.to_thread(_fetch_model_by_name_sync, model_name)


async def fetch_eval_session_stats() -> Dict[str, Any]:
    """eval_session_plan 상태 집계 async 래핑"""
    return await asyncio.to_thread(_fetch_eval_session_stats_sync)


# ===== MCP Tools 정의 =====
@server.list_tools()
async def list_tools() -> ListToolsResult:
    tools = [
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


# ===== MCP Tool 호출 처리 =====
@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    try:
        # 1) ai_model 전체 리스트
        if name == "list_ai_models":
            only_used = bool(arguments.get("only_used", False))
            models = await fetch_all_models(only_used=only_used)

            text = json.dumps(models, ensure_ascii=False, indent=2)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"ai_model 목록:\n{text}",
                    )
                ]
            )

        # 2) model_name으로 ai_model 단건 조회
        elif name == "get_ai_model_by_name":
            model_name = arguments.get("model_name")
            if not model_name:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="오류: model_name 인자가 필요합니다.",
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
                content=[
                    TextContent(
                        type="text",
                        text=f"AI 모델 정보:\n{text}",
                    )
                ]
            )

        # 3) eval_session_plan 상태별 카운트
        elif name == "get_eval_session_stats":
            stats = await fetch_eval_session_stats()
            text = json.dumps(stats, ensure_ascii=False, indent=2)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"eval_session_plan 상태 집계:\n{text}",
                    )
                ]
            )

        # 등록되지 않은 툴
        else:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"오류: '{name}' 툴을 찾을 수 없습니다.",
                    )
                ],
                isError=True,
            )

    except Exception as e:
        logger.exception("툴 실행 중 오류 발생")
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"툴 실행 중 오류 발생: {str(e)}",
                )
            ],
            isError=True,
        )


async def main():
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

        await server.run(
            read_stream,
            write_stream,
            init_options,
        )

    logger.info("🛑 MCP 서버 종료")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ MCP 서버 실행 중 오류 발생: {e}", exc_info=True)
