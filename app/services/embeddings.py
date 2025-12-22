# app/services/embeddings.py
from typing import List
import numpy as np
import httpx
from app.core.config import get_settings  # Settings 정의된 위치에 맞게 수정!
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaEmbeddingService:
    """Ollama /api/embed 엔드포인트를 이용하는 임베딩 서비스"""

    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.timeout = settings.OLLAMA_TIMEOUT
        self._dimension = None

    def _get_embedding_model(self) -> str:
        """DB에서 최신 임베딩 모델 설정을 가져옵니다."""
        from app.db.database import VectorSessionLocal
        from app.crud.config import get_app_config

        try:
            with VectorSessionLocal() as db:
                config = get_app_config(db)
                if config and config.embedding_model:
                    return config.embedding_model
        except Exception as e:
            logger.error(f"DB에서 임베딩 모델 정보를 가져오는데 실패했습니다: {e}")

        # 기본값 (DB 연결 실패 시 등)
        return "nomic-embed-text"

    @property
    def device(self) -> str:
        return "ollama + gpu"  # Ollama는 내부적으로 GPU 사용

    @property
    def dimension(self) -> int:
        """임베딩 벡터 차원 수 (한 번만 계산해서 캐시)"""
        if self._dimension is None:
            vecs = self._post_embeddings(["dim_probe"])
            self._dimension = int(vecs.shape[1])
        return self._dimension

    def _post_embeddings(self, inputs: List[str]) -> np.ndarray:
        """
        Ollama /api/embed 호출해서 임베딩 벡터 받아오기
        입력: 문자열 리스트
        출력: (len(inputs), dim) numpy 배열
        """
        url = f"{self.base_url}/api/embed"
        model = self._get_embedding_model()

        payload = {
            "model": model,
            "input": inputs,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        emb_list = data.get("embeddings")
        if emb_list is None:
            raise RuntimeError(f"Ollama 응답에 'embeddings' 키가 없습니다: {data}")

        arr = np.asarray(emb_list, dtype="float32")
        return arr

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self._post_embeddings(texts)

    def embed_query(self, query: str) -> np.ndarray:
        vecs = self._post_embeddings([query])
        return vecs


# 전역에서 쓸 싱글톤 인스턴스
embedding_service = OllamaEmbeddingService()
