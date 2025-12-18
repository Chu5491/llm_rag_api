# app/models/vector_models.py
from sqlalchemy import Column, Integer, Text, JSON, DateTime
from pgvector.sqlalchemy import Vector
from app.db.database import Base
from datetime import datetime


class FileEmbedding(Base):
    __tablename__ = "file_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))  # nomic-embed-text 기준 1024차원
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class FigmaEmbedding(Base):
    __tablename__ = "figma_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
