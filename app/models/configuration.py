from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime
from app.db.database import VectorBase


class Configuration(VectorBase):
    """
    시스템 전역 설정을 저장하는 테이블 (configurations).
    """

    __tablename__ = "configurations"

    id = Column(Integer, primary_key=True, index=True)

    # [Model Configuration]
    embedding_model = Column(String(255), default="bona/bge-m3-korean")
    llm_model = Column(String(255))

    # [RAG Settings]
    rag_tc_count = Column(Integer, default=20)
    rag_batch_size = Column(Integer, default=3)
    rag_tc_id_prefix = Column(String(50), default="REQ_TC")

    # [Figma Settings]
    figma_enabled = Column(Boolean, default=True)

    # [Meta Info]
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
