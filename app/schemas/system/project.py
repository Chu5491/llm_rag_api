from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from uuid import UUID
from datetime import datetime


class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    """
    CREATE TABLE projects의 insert 입력용
    - id/created_at/updated_at는 DB가 채움
    """
    pass


class ProjectUpdate(BaseModel):
    """
    부분 수정(PATCH)용
    - None이면 미변경
    """
    name: Optional[str] = Field(default=None, min_length=1)
    description: Optional[str] = None


class ProjectInDB(ProjectBase):
    """
    DB에서 읽어온 projects 레코드 응답용
    """
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)  # ORM 대응(예: SQLAlchemy)
