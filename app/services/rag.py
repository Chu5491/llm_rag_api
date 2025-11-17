# -*- coding: utf-8 -*-
# 각 줄 위에 주석 스타일, 탭 들여쓰기 유지

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

logger = logging.getLogger("rag")


class RAGManager:
	"""
	RAG용 문서 색인/검색 매니저
	"""

	SEARCH_MODES: Dict[str, Dict[str, object]] = {
		"fast": {
			"description": "빠른 검색 - 핵심만 추출",
			"search_type": "similarity",
			"k": 5,
			"score_threshold": 0.55
		},
		"balanced": {
			"description": "균형 잡힌 검색 (default)",
			"search_type": "similarity_score_threshold",
			"k": 15,
			"score_threshold": 0.6
		},
		"comprehensive": {
			"description": "포괄적 검색",
			"search_type": "mmr",
			"k": 30,
			"fetch_k": 50,
			"lambda_mult": 0.7
		}
	}

	PROMPT_TEMPLATE = """다음 문서들을 참고하여 질문에 답해주세요.
제공된 문서에 없는 내용은 절대 추측하지 말아주세요.

문서:
{context}

질문:
{question}

답변 (한국어로 간결하게):"""

	def __init__(
		self,
		documents_dir: str = "uploads",
		index_dir: str = "faiss_index",
		embedding_model: str = "jhgan/ko-sroberta-multitask",
	):
		self.documents_dir = Path(documents_dir)
		self.index_dir = Path(index_dir)
		self.embedding_model_name = embedding_model
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.embeddings: Optional[HuggingFaceEmbeddings] = None
		self.vector_store: Optional[FAISS] = None
		self.chunk_count = 0
		self.last_build_duration = 0.0
		self.default_ollama_model = "gpt-oss:20b"
		self._lock = threading.Lock()

	def _load_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
		target = model_name or self.embedding_model_name
		if self.embeddings and self.embedding_model_name == target:
			return self.embeddings

		logger.info("🔥 임베딩 모델 로드: %s on %s", target, self.device)
		self.embedding_model_name = target
		self.embeddings = HuggingFaceEmbeddings(
			model_name=self.embedding_model_name,
			model_kwargs={"device": self.device},
			encode_kwargs={"normalize_embeddings": True}
		)
		return self.embeddings

	def _index_exists_on_disk(self) -> bool:
		return self.index_dir.exists() and any(self.index_dir.glob("*.faiss"))

	def _load_raw_documents(self) -> List[Document]:
		if not self.documents_dir.exists():
			raise FileNotFoundError(f"문서 디렉토리를 찾을 수 없습니다: {self.documents_dir}")

		loaders = [
			DirectoryLoader(self.documents_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
			DirectoryLoader(self.documents_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8", "errors": "ignore"}),
			DirectoryLoader(self.documents_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8", "errors": "ignore"}),
		]

		documents: List[Document] = []
		for loader in loaders:
			try:
				docs = loader.load()
				logger.info("🗂️ %d개 문서 로드 (%s)", len(docs), loader.glob)
				documents.extend(docs)
			except Exception as exc:
				logger.warning("로더 오류 (%s): %s", loader.glob, exc)

		if not documents:
			raise ValueError("문서를 찾을 수 없습니다. uploads 디렉토리에 파일을 넣어주세요.")

		return documents

	def _split_documents(self, documents: List[Document]) -> List[Document]:
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=800,
			chunk_overlap=100,
			separators=["\n\n", "\n", ".", "!", "?", " ", ""],
		)
		return text_splitter.split_documents(documents)

	def _build_new_index(self, model_name: Optional[str] = None) -> int:
		embeddings = self._load_embeddings(model_name)
		raw_documents = self._load_raw_documents()
		chunks = self._split_documents(raw_documents)

		start = time.monotonic()
		self.vector_store = FAISS.from_documents(chunks, embeddings)
		duration = time.monotonic() - start

		self.chunk_count = len(chunks)
		self.last_build_duration = duration
		self.index_dir.mkdir(parents=True, exist_ok=True)
		self.vector_store.save_local(str(self.index_dir))

		logger.info("💾 벡터 DB 저장: %s (%d개 청크, %.1fs)", self.index_dir, self.chunk_count, duration)
		return self.chunk_count

	def ensure_vector_store(self, force: bool = False, embedding_model: Optional[str] = None) -> int:
		target_model = embedding_model or self.embedding_model_name
		need_rebuild = target_model != self.embedding_model_name

		with self._lock:
			if not force and self.vector_store and not need_rebuild:
				return self.chunk_count

			if not force and not need_rebuild and self._index_exists_on_disk():
				try:
					embeddings = self._load_embeddings(target_model)
					self.vector_store = FAISS.load_local(
						str(self.index_dir),
						embeddings,
						allow_dangerous_deserialization=True
					)
					self.chunk_count = int(self.vector_store.index.ntotal)
					logger.info("📂 기존 벡터 DB 로드: %s (%d개 청크)", self.index_dir, self.chunk_count)
					return self.chunk_count
				except Exception as exc:
					logger.warning("기존 벡터 DB 로드 실패: %s", exc)

			return self._build_new_index(target_model)

	def retrieve_documents(self, question: str, search_mode: Optional[str] = None) -> Tuple[List[Document], Dict[str, object]]:
		if not self.vector_store:
			raise RuntimeError("벡터 DB가 준비되지 않았습니다.")

		mode = (search_mode or "balanced").strip().lower()
		config = self.SEARCH_MODES.get(mode, self.SEARCH_MODES["balanced"])

		retriever = self.vector_store.as_retriever(
			search_type=config["search_type"],
			search_kwargs={k: v for k, v in config.items() if k not in {"search_type", "description"}}
		)

		docs = retriever.get_relevant_documents(question)
		return docs[: config["k"]], {"mode": mode, **config}

	def compose_context(self, docs: List[Document], max_chars: int = 3600) -> Tuple[str, List[str]]:
		parts: List[str] = []
		sources: List[str] = []
		total = 0
		for idx, doc in enumerate(docs):
			text = doc.page_content.strip()
			if not text:
				continue
			source = doc.metadata.get("source", f"chunk-{idx+1}")
			snippet = " ".join(text.splitlines())
			if len(snippet) > 1000:
				snippet = snippet[:1000] + "..."

			entry = f"[{idx+1}] {source}\n{snippet}"
			parts.append(entry)
			sources.append(f"{source} | {snippet[:200]}{'...' if len(snippet) > 200 else ''}")

			total += len(entry)
			if total >= max_chars:
				break

		return "\n\n".join(parts), sources

	def build_prompt(self, context: str, question: str) -> str:
		return self.PROMPT_TEMPLATE.format(context=context, question=question.strip())

	def status(self) -> Dict[str, object]:
		return {
			"ready": self.vector_store is not None,
			"chunk_count": self.chunk_count,
			"index_path": str(self.index_dir.resolve()) if self.index_dir.exists() else None,
			"embedding_model": self.embedding_model_name,
			"device": self.device,
			"last_build_duration": self.last_build_duration,
			"search_modes": list(self.SEARCH_MODES.keys()),
		}


rag_manager = RAGManager()
