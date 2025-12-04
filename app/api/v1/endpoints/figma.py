# FastAPI 라우터 임포트
from fastapi import APIRouter, Depends

# HTTP 예외 처리를 위한 임포트
from fastapi import HTTPException
import httpx

# 설정 의존성 임포트
from app.api.deps import get_app_settings

# Figma 클라이언트 임포트
from app.services.figma_client import FigmaClient

# 스키마 임포트
from app.schemas.figma import (
    FigmaProjectInfo,
    FigmaProjectFilesResponse,
)

from app.core.logging import get_logger

# figma parser 임포트
from app.utils.figma_url_parser import parse_figma_team_project

# typing 임포트
from typing import List

Settings = get_app_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/figma", tags=["figma"])


# 상태 조회 엔드포인트
@router.get("/info", response_model=FigmaProjectInfo)
async def get_figma_info():
    # 상태 조회 실행
    data = parse_figma_team_project(Settings.FIGMA_URL)
    # 결과 반환
    return FigmaProjectInfo(**data)


@router.get("/projects/files", response_model=FigmaProjectFilesResponse)
async def get_project_files(branch_data: bool = False):
    """프로젝트의 파일 목록을 가져옵니다."""
    client = FigmaClient()

    data = parse_figma_team_project(Settings.FIGMA_URL)
    logger.info(f"피그마 접속 URL 파싱: {data}")

    try:
        result = await client.get_project_files(data.get("project_id"), branch_data)
        logger.info(f"피그마 프로젝트 내 파일 수 : {len(result.files)}")
        return result
    except httpx.HTTPStatusError as e:
        logger.error(f"피그마 API error: {e}")
        logger.error(f"Response text: {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code, detail=f"피그마 API error: {str(e)}"
        )
