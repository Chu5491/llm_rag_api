# 설정 관리를 위한 BaseSettings 임포트
from pydantic_settings import BaseSettings

# 애플리케이션 전역 설정 클래스
class Settings(BaseSettings):
	# Ollama 서버 베이스 URL
	OLLAMA_BASE_URL: str = "http://localhost:11434"

	# Ollama HTTP 타임아웃(초)
	OLLAMA_TIMEOUT: int = 10

	# 임베딩 모델 이름 (예: 'nomic-embed-text', 'bona/bge-m3-korean' 등)
	EMBEDDING_MODEL: str=  "nomic-embed-text"

	# LLM 모델 (chat/generate용) ← 새로 추가
	LLM_MODEL: str = "llama3.1:8b"
	
	# rag 대상 파일 경로
	UPLOADS_DIR: str = "uploads"

	# FAISS 인덱스(벡터) 파일 경로
	VECTOR_INDEX_PATH: str = "data/index.faiss"

	# 메타데이터(JSON) 파일 경로
	VECTOR_META_PATH: str = "data/index_meta.json"

	# 환경변수 로딩 파일 지정(.env)
	class Config:
		# dotenv 사용 허용
		env_file = ".env"
		# 케이스 민감도 설정
		case_sensitive = True

# 싱글톤 성격의 설정 인스턴스 함수
def get_settings() -> Settings:
	# Settings 인스턴스 생성 및 반환
	return Settings()
