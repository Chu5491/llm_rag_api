# 설정 의존성 주입을 위한 함수 임포트
from app.core.config import get_settings, Settings

# FastAPI Depends에서 사용할 팩토리
def get_app_settings() -> Settings:
	# 앱 설정 인스턴스 반환
	return get_settings()
