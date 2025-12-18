# FastAPI 라우터 임포트
from fastapi import APIRouter

# HTTP 예외 처리를 위한 임포트
from fastapi import HTTPException
import httpx

# 설정 의존성 임포트
from app.api.deps import get_app_settings

# Figma 클라이언트 임포트
from app.services.figma_client import FigmaClient

# 스키마 임포트
from app.schemas.figma import (
    FigmaProjectFilesResponse,
)

from app.core.logging import get_logger

# figma parser 임포트
from app.utils.figma_url_parser import parse_figma_team_project

Settings = get_app_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/figma", tags=["figma"])


@router.get("/info", response_model=FigmaProjectFilesResponse)
async def get_pat_project_info(branch_data: bool = False):
    """프로젝트의 파일 목록을 가져오며, 이 호출이 성공하면 연동 가능으로 본다."""
    client = FigmaClient()

    data = parse_figma_team_project(Settings.FIGMA_URL)
    logger.info(f"피그마 접속 URL 파싱: {data}")

    project_id = data.get("project_id")
    if not project_id:
        raise HTTPException(
            status_code=400,
            detail="FIGMA_URL에서 project_id를 파싱하지 못했습니다.",
        )

    try:
        # 연동 가능 여부 체크용 호출
        result = await client.get_project_files(project_id, branch_data)
        logger.info(f"피그마 프로젝트 내 파일 수 : {len(result.files)}")

        return result

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code

        # Figma가 내려주는 원본 메시지도 한 번 빼두고
        try:
            err_json = e.response.json()
            figma_message = err_json.get("message") or e.response.text
        except ValueError:
            figma_message = e.response.text

        logger.error(f"피그마 API error: {e}")
        logger.error(f"Response text: {figma_message}")

        if status_code in (401, 403):
            detail = "Figma PAT가 올바르지 않거나 해당 프로젝트에 접근 권한이 없습니다."
        elif status_code == 404:
            detail = "Figma 프로젝트를 찾을 수 없습니다. FIGMA_URL 또는 프로젝트 권한을 확인하세요."
        else:
            detail = f"피그마 API error({status_code}): {figma_message}"

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        )
