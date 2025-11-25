# app/api/v1/endpoints/mcp_server.py
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")

# 예시 DB 흉내 내기
FAKE_USERS = {
    "chu": {
        "id": "chu",
        "name": "추사용자",
        "email": "chu@example.com",
    },
    "test": {
        "id": "test",
        "name": "테스트 유저",
        "email": "test@example.com",
    },
}

async def get_user_by_id(user_id: str) -> Dict[str, Any] | None:
    return FAKE_USERS.get(user_id)

@mcp.tool()
async def get_user(user_id: str) -> Dict[str, Any]:
    """
    유저 ID로 유저 정보를 조회하는 MCP Tool.
    """
    user = await get_user_by_id(user_id)

    if user is None:
        return {
            "found": False,
            "message": f"ID가 '{user_id}'인 유저를 찾을 수 없어요.",
        }

    return {
        "found": True,
        "user": user,
        "message": f"ID '{user_id}' 유저 정보를 가져왔어요.",
    }

def main():
    # STDIO 기반 MCP 서버로 동작
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
