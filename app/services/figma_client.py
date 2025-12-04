import httpx
from typing import List, Dict, Any
from app.core.config import get_settings
from app.schemas.figma import FigmaProjectFilesResponse


class FigmaClient:
    """Figma API 클라이언트"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.figma.com/v1"
        self.headers = {"X-Figma-Token": self.settings.FIGMA_API_TOKEN}

    async def get_project_files(
        self, project_id: str, branch_data: bool = False
    ) -> FigmaProjectFilesResponse:
        """프로젝트의 파일 목록을 가져옵니다."""
        url = f"{self.base_url}/projects/{project_id}/files"
        params = {"branch_data": str(branch_data).lower()}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return FigmaProjectFilesResponse(**response.json())
