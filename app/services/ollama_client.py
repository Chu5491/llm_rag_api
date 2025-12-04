# 비동기 HTTP 클라이언트
import httpx
# 타입 힌트용 프로토콜
from typing import Any, Dict, List, Optional
# 설정 타입 임포트
from app.core.config import Settings

# Ollama API 클라이언트
class OllamaClient:
	# 생성자에서 설정을 주입받음
	def __init__(self, settings: Settings):
		# 베이스 URL 보관
		self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
		# 타임아웃 보관
		self.timeout = settings.OLLAMA_TIMEOUT

	# 내부용 비동기 클라이언트 생성
	def _client(self) -> httpx.AsyncClient:
		# 베이스 URL과 타임아웃이 적용된 클라이언트 반환
		return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

	# 서버 버전/상태 조회
	async def get_status(self) -> Dict[str, Any]:
		# 비동기 클라이언트 컨텍스트 오픈
		async with self._client() as client:
			# /api/version 엔드포인트 호출
			resp = await client.get("/api/version")
			# 실패 시 예외 발생
			resp.raise_for_status()
			# JSON 반환
			data = resp.json()
			# 버전 키가 없을 수도 있으니 보완 정보 추가
			return {
				"reachable": True,
				"version": data.get("version"),
				"raw": data
			}
	
	# 모델 리스트 조회
	async def list_models(self) -> Dict[str, Any]:
		# 비동기 클라이언트 컨텍스트 오픈
		async with self._client() as client:
			# /api/tags 엔드포인트 호출
			resp = await client.get("/api/tags")
			# 실패 시 예외 발생
			resp.raise_for_status()
			# JSON 반환
			return resp.json()

	# LLM chat 호출 (다중 메시지 - MCP용)
	async def chat_with_messages(
		self,
		messages: List[Dict[str, str]],
		model: Optional[str] = None,
		stream: bool = False,
		options: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		async with self._client() as client:
			payload = {
				"model": model,
				"messages": messages,
				"stream": stream,
			}
			if options is not None:
				payload["options"] = options

			resp = await client.post("/api/chat", json=payload)
			resp.raise_for_status()
			return resp.json()
