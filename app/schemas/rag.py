# app/schemas/rag.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from enum import Enum

class EmbedDebugResponse(BaseModel):
    """임베딩 디버그용 응답"""
    device: str
    dimension: int
    vector_length: int

class RagQARequest(BaseModel):
    """RAG QA 요청 바디"""
    model: str | None = None  # 없으면 설정값 사용


class RagQAResponse(BaseModel):
    """RAG QA 응답 바디"""
    answer: str
    contexts: List[Dict[str, Any]]

""" DB """
class RagSource(str, Enum):
    """
    RAG 청크의 출처를 나타내는 열거형
    
    Attributes:
        FILE: 로컬 파일 시스템 또는 업로드된 파일에서 추출한 청크
        FIGMA: Figma 디자인 문서에서 추출한 청크
        JIRA: JIRA 이슈에서 추출한 청크
    """
    FILE = "FILE"
    FIGMA = "FIGMA"
    JIRA = "JIRA"


class RagChunkBase(BaseModel):
    """
    RAG(Retrieval-Augmented Generation)을 위한 청크의 기본 모델
    
    모든 RAG 청크는 이 클래스를 상속받아 사용됩니다.
    
    Attributes:
        project_id: 청크가 속한 프로젝트의 고유 ID
        source: 청크의 출처 (RagSource 참조)
        file_id: 파일 기반 청크의 경우 파일 ID (source=FILE일 때 사용)
        external_key: 외부 시스템에서의 고유 키 (source=FIGMA/JIRA일 때 사용)
        chunk_index: 동일 출처 내에서의 청크 순서 (0부터 시작)
        content: 청크의 텍스트 내용 (최소 1자 이상 필요)
        metadata: 청크와 관련된 추가 메타데이터 (키-값 쌍)
        embedding_model: 이 청크의 임베딩을 생성하는 데 사용된 모델 이름
    """
    project_id: UUID
    source: RagSource

    # FILE이면 file_id 사용,
	# FIGMA/JIRA면 external_key 사용
    file_id: Optional[UUID] = None
    external_key: Optional[str] = None

    chunk_index: int = 0
    content: str = Field(..., min_length=1)

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="청크와 관련된 추가 메타데이터. 예: 페이지 번호, 섹션 제목 등"
    )
    embedding_model: str = Field(
        ...,
        min_length=1,
        description="이 청크의 임베딩을 생성하는 데 사용된 모델의 이름 (예: 'sentence-transformers/all-MiniLM-L6-v2')"
    )


class RagChunkCreate(RagChunkBase):
    """
    생성용
    - 보통 embedding은 서버에서 생성하니까 요청에서는 제외하는 걸 추천
    """
    pass


class RagChunkCreateWithEmbedding(RagChunkBase):
    """
    내부/배치용(이미 embedding을 만들어서 넣는 경우)
    """
    embedding: List[float]


class RagChunkUpdate(BaseModel):
    """
    PATCH용 (원본은 보통 바꾸지 않고 content/metadata만 교체하거나 재임베딩)
    """
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding_model: Optional[str] = None
    embedding: Optional[List[float]] = None


class RagChunkInDB(BaseModel):
    """
    데이터베이스에 저장된 RAG 청크 정보를 나타내는 모델
    
    Attributes:
        id: 청크의 고유 식별자
        project_id: 청크가 속한 프로젝트의 ID
        source: 청크의 출처 (FILE, FIGMA, JIRA)
        file_id: 파일 기반 청크인 경우 파일 ID (source가 FILE일 때 사용)
        external_key: 외부 시스템에서의 고유 키 (source가 FIGMA나 JIRA일 때 사용)
        chunk_index: 청크의 순차적 인덱스
        content: 청크의 텍스트 내용
        metadata: 청크와 관련된 추가 메타데이터
        embedding_model: 임베딩 생성에 사용된 모델 이름
        created_at: 청크 생성 일시
    """
    id: UUID
    project_id: UUID
    source: RagSource
    file_id: Optional[UUID]
    external_key: Optional[str]
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    embedding_model: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
    """
    Pydantic 모델 설정
    - from_attributes=True: ORM 모델에서의 자동 변환을 허용 (SQLAlchemy 등과 호환)
    """