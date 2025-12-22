from sqlalchemy import Column, Integer, String, Text, DateTime, Interval
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone
from app.db.database import VectorBase


class GenerateHistory(VectorBase):
    """
    테스트케이스 자동 생성 실행 이력 테이블 (generate_history)
    """

    __tablename__ = "generate_history"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, default=1, nullable=False)
    title = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)
    status = Column(String(20), default="running", nullable=False)
    summary = Column(Text, nullable=True)

    total_batches = Column(Integer, default=0)
    current_batch = Column(Integer, default=0)
    progress = Column(Integer, default=0)

    started_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    finished_at = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Interval, nullable=True)

    logs = Column(Text, nullable=True)
    result_data = Column(JSONB, nullable=True)
    model_name = Column(String(100), nullable=True)
