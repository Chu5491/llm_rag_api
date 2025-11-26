#!/usr/bin/env python3
"""
MCP 서버 - 더미 데이터를 제공하는 테스트 서버
"""

import asyncio
import json
from typing import Any, Dict
from mcp.server import Server, InitializationOptions, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Tool,
    TextContent,
)

# MCP 서버 인스턴스 생성
server = Server("dummy-mcp-server")

# 더미 데이터베이스
USERS_DB = {
    "user1": {
        "id": "user1",
        "name": "김철수",
        "email": "kim@example.com",
        "department": "개발팀",
        "join_date": "2023-01-15"
    },
    "user2": {
        "id": "user2", 
        "name": "이영희",
        "email": "lee@example.com",
        "department": "기획팀",
        "join_date": "2022-03-20"
    },
    "user3": {
        "id": "user3",
        "name": "박민준",
        "email": "park@example.com", 
        "department": "디자인팀",
        "join_date": "2023-06-10"
    }
}

PROJECTS_DB = [
    {
        "id": "proj1",
        "name": "LLM RAG 시스템",
        "status": "진행중",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "team": ["user1", "user2"]
    },
    {
        "id": "proj2", 
        "name": "MCP 통합 프로젝트",
        "status": "계획중",
        "start_date": "2024-07-01",
        "end_date": "2024-09-30",
        "team": ["user1", "user3"]
    }
]

@server.list_tools()
async def list_tools() -> ListToolsResult:
    """사용 가능한 툴 목록 반환"""
    tools = [
        Tool(
            name="get_user",
            description="특정 사용자 정보를 조회합니다. user_id를 인자로 받습니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "조회할 사용자의 ID (예: user1, user2, user3)"
                    }
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="list_all_users",
            description="모든 사용자 목록을 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_project",
            description="특정 프로젝트 정보를 조회합니다. project_id를 인자로 받습니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string", 
                        "description": "조회할 프로젝트의 ID (예: proj1, proj2)"
                    }
                },
                "required": ["project_id"]
            }
        ),
        Tool(
            name="list_all_projects",
            description="모든 프로젝트 목록을 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="search_users_by_department",
            description="부서별 사용자를 검색합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "검색할 부서명 (예: 개발팀, 기획팀, 디자인팀)"
                    }
                },
                "required": ["department"]
            }
        ),
        Tool(
            name="get_current_time",
            description="현재 시스템 시간을 반환합니다.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]
    return ListToolsResult(tools=tools)

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """툴 호출 처리"""
    
    try:
        if name == "get_user":
            user_id = arguments.get("user_id")
            if not user_id:
                return CallToolResult(
                    content=[TextContent(type="text", text="오류: user_id 인자가 필요합니다.")],
                    isError=True
                )
            
            user = USERS_DB.get(user_id)
            if user:
                return CallToolResult(
                    content=[TextContent(
                        type="text", 
                        text=f"사용자 정보:\n{json.dumps(user, ensure_ascii=False, indent=2)}"
                    )]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"오류: '{user_id}' 사용자를 찾을 수 없습니다.")],
                    isError=True
                )
        
        elif name == "list_all_users":
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"전체 사용자 목록:\n{json.dumps(list(USERS_DB.values()), ensure_ascii=False, indent=2)}"
                )]
            )
        
        elif name == "get_project":
            project_id = arguments.get("project_id")
            if not project_id:
                return CallToolResult(
                    content=[TextContent(type="text", text="오류: project_id 인자가 필요합니다.")],
                    isError=True
                )
            
            project = next((p for p in PROJECTS_DB if p["id"] == project_id), None)
            if project:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"프로젝트 정보:\n{json.dumps(project, ensure_ascii=False, indent=2)}"
                    )]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"오류: '{project_id}' 프로젝트를 찾을 수 없습니다.")],
                    isError=True
                )
        
        elif name == "list_all_projects":
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"전체 프로젝트 목록:\n{json.dumps(PROJECTS_DB, ensure_ascii=False, indent=2)}"
                )]
            )
        
        elif name == "search_users_by_department":
            department = arguments.get("department")
            if not department:
                return CallToolResult(
                    content=[TextContent(type="text", text="오류: department 인자가 필요합니다.")],
                    isError=True
                )
            
            filtered_users = [
                user for user in USERS_DB.values() 
                if user["department"] == department
            ]
            
            if filtered_users:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"'{department}' 부서 사용자 목록:\n{json.dumps(filtered_users, ensure_ascii=False, indent=2)}"
                    )]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"'{department}' 부서에 속한 사용자가 없습니다.")],
                    isError=True
                )
        
        elif name == "get_current_time":
            import datetime
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return CallToolResult(
                content=[TextContent(type="text", text=f"현재 시간: {current_time}")]
            )
        
        else:
            return CallToolResult(
                content=[{"type": "text", "text": f"오류: '{name}' 툴을 찾을 수 없습니다."}],
                isError=True
            )
            
    except Exception as e:
        return CallToolResult(
            content=[{"type": "text", "text": f"툴 실행 중 오류 발생: {str(e)}"}],
            isError=True
        )

async def main():
    """MCP 서버 실행
    print("[MCP Server] 더미 MCP 서버 시작...")
    print("[MCP Server] 사용 가능한 툴:")
    print("  - get_user: 사용자 정보 조회")
    print("  - list_all_users: 전체 사용자 목록")
    print("  - get_project: 프로젝트 정보 조회") 
    print("  - list_all_projects: 전체 프로젝트 목록")
    print("  - search_users_by_department: 부서별 사용자 검색")
    print("  - get_current_time: 현재 시간 조회")
    """
    
    async with stdio_server() as (read_stream, write_stream):
        init_options = InitializationOptions(
            server_name="dummy-mcp-server",
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

if __name__ == "__main__":
    asyncio.run(main())
