# app/models/vector_models.py
from sqlalchemy import Column, Integer, Text, JSON, DateTime, String
from pgvector.sqlalchemy import Vector
from app.db.database import VectorBase
from datetime import datetime


class RagEmbedding(VectorBase):
    """
    모든 소스(File, Figma, Jira 등)에서 생성된 임베딩을 통합 관리하는 테이블
    """

    __tablename__ = "rag_embeddings"
    id = Column(Integer, primary_key=True, index=True)

    # 나중에 생길 project 테이블의 id(PK)와 타입을 맞춤 (Integer)
    # TODO:현재는 프로젝트 기능이 없으므로 1을 기본값으로 사용
    project_id = Column(Integer, nullable=False, index=True, default=1)

    source_type = Column(String(50), nullable=False, index=True)  # 'file', 'figma' 등
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
