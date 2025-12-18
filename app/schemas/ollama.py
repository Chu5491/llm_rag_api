# Pydantic BaseModel 임포트
from pydantic import BaseModel
# 타입 힌트용
from typing import Any, List, Literal, Optional, Dict

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

# 메시지 객체
class OllamaChatMessage(BaseModel):
	role: Literal["user", "assistant", "system"]
	content: str
	
# Ollama chat 요청 스키마
class OllamaChatRequest(BaseModel):
	# 프롬프트 텍스트
	messages: List[OllamaChatMessage]
	
	# 사용할 모델명 (예: "llama3") - 설정 기본값 쓰고 싶으면 옵션
	model: Optional[str] = None
	
	# 스트리밍 여부 (기본 false)
	stream: bool = False
	
	# Ollama options (온도, 반복 패널티 등)
	options: Optional[Dict[str, Any]] = None

# Ollama chat 응답 스키마
class OllamaChatResponse(BaseModel):
	# 호출 성공 여부
	success: bool = True
	
	# 생성된 텍스트 (에러 시 None)
	output: Optional[str] = None
	
	# 원시 응답 그대로 (디버깅용)
	raw: Optional[Dict[str, Any]] = None