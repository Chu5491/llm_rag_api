# 각 줄 위에 주석 스타일, 탭 들여쓰기 유지
# 설정 관리를 위한 BaseSettings 임포트
from pydantic_settings import BaseSettings
# 타입 힌트용 모듈 임포트
from typing import Optional

# 애플리케이션 전역 설정 클래스
class Settings(BaseSettings):
	# Ollama 서버 베이스 URL
	OLLAMA_BASE_URL: str = "http://localhost:11434"
	# Ollama HTTP 타임아웃(초)
	OLLAMA_TIMEOUT: int = 10

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
