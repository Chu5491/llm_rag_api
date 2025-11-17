# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RagBuildRequest(BaseModel):
	force: bool = Field(False, description="기존 색인을 강제로 재생성할지 여부")
	embedding_model: Optional[str] = Field(None, description="사용하려는 임베딩 모델명 (기본 jhgan/ko-sroberta-multitask)")


class RagBuildResponse(BaseModel):
	success: bool
	chunk_count: int
	message: str
	duration: float = Field(..., description="마지막 색인 생성 소요 시간(초)")


class RagStatusResponse(BaseModel):
	ready: bool
	chunk_count: int
	index_path: Optional[str]
	embedding_model: str
	device: str
	last_build_duration: float
	search_modes: List[str]


class RagQueryRequest(BaseModel):
	question: str = Field(..., description="질문 텍스트")
	search_mode: Optional[str] = Field(None, description="fast / balanced / comprehensive 중 하나")
	model: Optional[str] = Field(None, description="Ollama 모델명 (기본 gpt-oss:20b)")


class RagQueryResponse(BaseModel):
	success: bool
	answer: Optional[str]
	sources: List[str]
	search_mode: str
	model: str
	response_time: float
	raw_response: Optional[Dict[str, Any]]
	context: Optional[str]
