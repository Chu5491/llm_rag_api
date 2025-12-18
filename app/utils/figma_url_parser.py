# utils/figma_url_parser.py
from urllib.parse import urlparse
from app.core.logging import get_logger
from app.schemas.figma import FigmaProjectInfo

logger = get_logger(__name__)


def parse_figma_team_project(url: str) -> FigmaProjectInfo:
    """
    Figma 팀/프로젝트 URL에서 team_id와 project_id를 파싱한다.

    예)
    https://www.figma.com/files/team/1111111111/project/2222222222/Team-project?fuid=...
      -> team_id = "1111111111"
         project_id = "2222222222"
    """
    parsed = urlparse(url)

    # 기본 형식 체크
    if "figma.com" not in parsed.netloc:
        raise ValueError(f"Figma URL이 아닌 것 같아요: {url}")

    # '/files/team/.../project/.../slug' 이런 식의 path를 분해
    parts = [p for p in parsed.path.split("/") if p]  # 빈 문자열 제거

    team_id = None
    project_id = None

    for i, part in enumerate(parts):
        if part == "team" and i + 1 < len(parts):
            team_id = parts[i + 1]
        if part == "project" and i + 1 < len(parts):
            project_id = parts[i + 1]

    if not team_id or not project_id:
        raise ValueError(
            f"team_id/project_id를 URL에서 찾지 못했습니다. path={parsed.path}"
        )

    return {"team_id": team_id, "project_id": project_id}
