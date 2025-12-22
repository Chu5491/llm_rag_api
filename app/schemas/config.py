from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AppConfigResponse(BaseModel):
    id: int
    embedding_model: Optional[str]
    llm_model: Optional[str]
    rag_tc_count: Optional[int]
    rag_batch_size: Optional[int]
    rag_tc_id_prefix: Optional[str]
    figma_enabled: Optional[bool]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AppConfigUpdate(BaseModel):
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    rag_tc_count: Optional[int] = None
    rag_batch_size: Optional[int] = None
    rag_tc_id_prefix: Optional[str] = None
    figma_enabled: Optional[bool] = None
