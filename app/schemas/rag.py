# app/schemas/rag.py

from typing import List, Dict, Any

from pydantic import BaseModel

class EmbedDebugResponse(BaseModel):
	"""임베딩 디버그용 응답"""
	device: str
	dimension: int
	vector_length: int

class RagQARequest(BaseModel):
	"""RAG QA 요청 바디"""
	query: str
	top_k: int = 4
	model: str | None = None  # 없으면 설정값 사용


class RagQAResponse(BaseModel):
	"""RAG QA 응답 바디"""
	answer: str
	contexts: List[Dict[str, Any]]
