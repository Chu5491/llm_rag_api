# app/services/rag_store.py

# 타입 힌트용
from typing import List, Dict, Any, Optional

# 파일 및 경로 작업용
import os
# JSON 저장/로드용
import json

# 숫자 벡터 연산용
import numpy as np
# PDF 읽기용
from pypdf import PdfReader
# FAISS 벡터 인덱스
import faiss

# 설정 가져오기
from app.core.config import get_settings
# Ollama 임베딩 서비스
from app.services.embeddings import embedding_service


class RagVectorStore:
	"""uploads 폴더 문서를 임베딩해서
	FAISS 인덱스로 관리하고 검색까지 담당하는 클래스"""

	def __init__(self):
		"""설정과 FAISS 인덱스 기본 상태를 초기화"""
		# 설정 로드
		self.settings = get_settings()

		# 임베딩 차원 수 (예: 1024)
		self.dimension: int = embedding_service.dimension

		# FAISS 인덱스 (처음엔 None)
		self.index: Optional[faiss.IndexFlatIP] = None

		# 인덱스에 들어간 각 벡터와 매핑되는 텍스트 / 메타데이터
		self.texts: List[str] = []
		self.metadatas: List[Dict[str, Any]] = []

	def _ensure_dirs(self) -> None:
		"""uploads, data 디렉토리가 없으면 생성"""
		# 업로드 디렉토리 생성
		os.makedirs(self.settings.UPLOADS_DIR, exist_ok=True)

		# 인덱스 / 메타데이터 파일이 들어갈 상위 디렉토리 생성
		index_dir = os.path.dirname(self.settings.VECTOR_INDEX_PATH)
		if index_dir:
			os.makedirs(index_dir, exist_ok=True)

	def _load_text_from_file(self, path: str) -> List[Dict[str, Any]]:
		"""단일 파일을 읽어 텍스트 청크와 메타데이터 리스트로 반환

		반환 형태 예:
		[
			{"text": "...", "meta": {"source": "a.pdf", "page": 0, "type": "pdf"}},
			...
		]
		"""
		results: List[Dict[str, Any]] = []

		# 확장자 소문자로
		ext = os.path.splitext(path)[1].lower()

		# 텍스트 파일인 경우(.txt, .md)
		if ext in [".txt", ".md"]:
			with open(path, "r", encoding="utf-8") as f:
				content = f.read()

			# 공백만 있는 경우는 제외
			if content.strip():
				results.append({
					"text": content,
					"meta": {
						"source": os.path.basename(path),
						"type": "text",
					},
				})
			return results

		# PDF 파일인 경우
		if ext == ".pdf":
			reader = PdfReader(path)
			for page_num, page in enumerate(reader.pages):
				text = page.extract_text() or ""
				if not text.strip():
					continue
				results.append({
					"text": text,
					"meta": {
						"source": os.path.basename(path),
						"type": "pdf",
						"page": page_num,
					},
				})
			return results

		# 그 외 포맷은 아직 지원 X
		return []

	def _build_from_uploads(self) -> None:
		"""uploads 디렉토리의 파일들로부터 새 인덱스를 생성"""
		# 디렉토리 준비
		self._ensure_dirs()

		all_texts: List[str] = []
		all_metas: List[Dict[str, Any]] = []

		# uploads 폴더 안의 파일들을 순회
		for filename in os.listdir(self.settings.UPLOADS_DIR):
			path = os.path.join(self.settings.UPLOADS_DIR, filename)

			# 디렉토리는 스킵
			if not os.path.isfile(path):
				continue

			# 파일에서 청크 리스트 가져오기
			entries = self._load_text_from_file(path)
			for entry in entries:
				all_texts.append(entry["text"])
				all_metas.append(entry["meta"])

		# 청크가 하나도 없으면 인덱스 생성 스킵
		if not all_texts:
			print("⚠️ uploads 디렉토리에 인덱싱할 문서가 없습니다.")
			return

		print(f"📄 총 {len(all_texts)} 개의 청크를 임베딩합니다...")

		# Ollama 임베딩으로 벡터 생성
		embeddings = np.ascontiguousarray(embedding_service.embed_texts(all_texts), dtype="float32")
		
        # 코사인 유사도 쓰기 위해 L2 정규화
		faiss.normalize_L2(embeddings)

		# FAISS 내적 기반 인덱스 생성
		index = faiss.IndexFlatIP(self.dimension)
		
		# 벡터 추가
		index.add(embeddings)

		# 메모리에 보관
		self.index = index
		self.texts = all_texts
		self.metadatas = all_metas

		# 디스크에 저장
		self._save_index()

	def _save_index(self) -> None:
		"""현재 인덱스와 메타데이터를 디스크에 저장"""
		# 인덱스가 없는 경우는 아무 것도 하지 않음
		if self.index is None:
			return

		# 디렉토리 준비
		self._ensure_dirs()

		# FAISS 인덱스 저장
		faiss.write_index(self.index, self.settings.VECTOR_INDEX_PATH)

		# 텍스트와 메타데이터를 JSON으로 저장
		meta_data = {
			"texts": self.texts,
			"metadatas": self.metadatas,
		}
		with open(self.settings.VECTOR_META_PATH, "w", encoding="utf-8") as f:
			json.dump(meta_data, f, ensure_ascii=False)

		print(f"💾 인덱스 저장 완료: {self.settings.VECTOR_INDEX_PATH}")

	def _load_index(self) -> None:
		"""디스크에서 인덱스와 메타데이터를 로드"""
		# 둘 중 하나라도 없으면 로드 불가
		if not (
			os.path.exists(self.settings.VECTOR_INDEX_PATH)
			and os.path.exists(self.settings.VECTOR_META_PATH)
		):
			print("⚠️ 인덱스 또는 메타데이터 파일이 없습니다. 새로 빌드해야 합니다.")
			return

		# FAISS 인덱스 로드
		index = faiss.read_index(self.settings.VECTOR_INDEX_PATH)

		# 메타데이터 로드
		with open(self.settings.VECTOR_META_PATH, "r", encoding="utf-8") as f:
			meta_data = json.load(f)

		self.index = index
		self.texts = meta_data.get("texts", [])
		self.metadatas = meta_data.get("metadatas", [])

		print(f"✅ 인덱스 로드 완료: {len(self.texts)} 개 청크")

	def ensure_vector_store(self) -> None:
		"""서버 시작 시 호출해서
		기존 인덱스가 있으면 로드, 없으면 uploads에서 새로 빌드"""
		# 이미 메모리에 인덱스가 있다면 스킵
		if self.index is not None:
			return

		# 인덱스/메타 파일이 둘 다 있으면 로드
		if os.path.exists(self.settings.VECTOR_INDEX_PATH) and os.path.exists(self.settings.VECTOR_META_PATH):
			print("📦 기존 인덱스를 로드합니다...")
			self._load_index()
		else:
			print("🚧 인덱스가 없으므로 uploads 디렉토리에서 새로 빌드합니다...")
			self._build_from_uploads()

	def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""쿼리 문자열을 임베딩하여 FAISS에서 top_k 유사한 청크를 반환"""
		# 인덱스가 준비되지 않았다면 예외
		if self.index is None:
			raise RuntimeError("벡터 인덱스가 초기화되지 않았습니다.")

		# 쿼리 임베딩
		query_vec = embedding_service.embed_query(query)
		
		# 정규화
		faiss.normalize_L2(query_vec)

		# FAISS 검색
		scores, indices = self.index.search(query_vec, top_k)
		scores = scores[0]
		indices = indices[0]

		results: List[Dict[str, Any]] = []

		for score, idx in zip(scores, indices):
			if idx < 0 or idx >= len(self.texts):
				continue
			results.append({
				"score": float(score),
				"text": self.texts[idx],
				"meta": self.metadatas[idx],
			})

		return results


# 앱 전체에서 공유해서 쓸 싱글톤 인스턴스
rag_vector_store = RagVectorStore()
