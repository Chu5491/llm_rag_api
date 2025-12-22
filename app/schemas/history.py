from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any


class HistoryResponse(BaseModel):
    id: int
    project_id: int
    title: str
    source_type: str
    status: str
    summary: Optional[str] = None
    total_batches: int
    current_batch: int
    progress: int
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration: Optional[Any] = None  # interval은 문자열이나 객체로 표현됨
    model_name: Optional[str] = None

    class Config:
        from_attributes = True


class HistoryDetailResponse(HistoryResponse):
    logs: Optional[str] = None
    result_data: Optional[Any] = None
