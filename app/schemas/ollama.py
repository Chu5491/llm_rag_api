# Pydantic BaseModel 임포트
from pydantic import BaseModel
# 타입 힌트용
from typing import Any, List, Optional, Dict

# Ollama 상태 응답 스키마
class OllamaStatus(BaseModel):
	# 서버 접근 가능 여부
	reachable: bool
	# Ollama 버전 문자열
	version: Optional[str] = None
	# 원본 응답(JSON)
	raw: Optional[Dict[str, Any]] = None

# Ollama 모델 하나의 스키마(간단화)
class OllamaModelItem(BaseModel):
	# 모델 이름
	name: str
	# 모델 다이제스트(있을 수 있음)
	digest: Optional[str] = None
	# 크기 등 기타 메타데이터
	size: Optional[int] = None

# Ollama 모델 리스트 응답 스키마
class OllamaModels(BaseModel):
	# 모델 배열(tags 필드 매핑)
	models: List[OllamaModelItem]
