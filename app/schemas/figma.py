from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# Figma 프로젝트 정보 스키마
class FigmaProjectInfo(BaseModel):
    team_id: str
    project_id: str


# 파일 정보 스키마
class FigmaFile(BaseModel):
    key: str
    name: str
    thumbnail_url: Optional[str] = None
    last_modified: datetime
    branch_data: Optional[dict] = None  # branch_data가 true일 경우 포함


# 프로젝트 파일 목록 응답 스키마
class FigmaProjectFilesResponse(BaseModel):
    name: str
    files: List[FigmaFile]
    cursor: Optional[dict] = None  # 페이지네이션을 위한 커서
