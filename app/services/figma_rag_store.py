# app/services/figma_rag_store.py

# 타입 힌트용
from typing import List, Dict, Any, Optional

# 파일 및 경로 작업용
import os

# JSON 저장/로드용
import json

# 숫자 벡터 연산용
import numpy as np


# FAISS 벡터 인덱스
import faiss

# DateTime
from datetime import datetime

# 비동기 처리용
import asyncio

# Ollama 임베딩 서비스
from app.services.embeddings import embedding_service

# 설정 가져오기
from app.core.config import get_settings

# 로깅
from app.core.logging import get_logger

# 피그마 관련
from app.schemas.figma import FigmaFileLLMSummary
from app.services.figma_client import FigmaClient
from app.utils.figma_url_parser import parse_figma_team_project

logger = get_logger(__name__)


class FigmaRagVectorStore:
    """
    FigmaFileLLMSummary를 임베딩해서
    Figma 전용 FAISS 인덱스로 관리하는 클래스
    """

    # Figma 전용 벡터 스토어를 초기화하고 설정/상태를 준비한다.
    def __init__(self):
        self.settings = get_settings()
        self.dimension: int = embedding_service.dimension

        # FAISS 인덱스 (처음엔 None)
        self.index: Optional[faiss.IndexFlatIP] = None

        # 인덱스에 들어간 각 벡터와 매핑되는 텍스트 / 메타데이터
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        # Figma 기능 활성화 여부 플래그
        self.enabled: bool = bool(self.settings.FIGMA_ENABLED)

        if self.enabled:
            logger.info(
                f"FigmaRagVectorStore 초기화: store_path={self.settings.FIGMA_VECTOR_INDEX_PATH}"
            )
        else:
            logger.info("Figma 비활성화 상태이므로 인덱스를 사용하지 않습니다.")

    # 인덱스 파일이 저장될 디렉토리가 없으면 생성한다.
    def _ensure_dirs(self) -> None:
        index_dir = os.path.dirname(self.settings.FIGMA_VECTOR_INDEX_PATH)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)

    # 현재 메모리의 Figma 인덱스와 메타데이터를 디스크에 저장한다.
    def _save_index(self) -> None:
        if self.index is None:
            return

        self._ensure_dirs()

        # FAISS 인덱스 저장
        faiss.write_index(self.index, self.settings.FIGMA_VECTOR_INDEX_PATH)

        # 텍스트와 메타데이터를 JSON으로 저장
        meta_data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
        }
        with open(self.settings.FIGMA_VECTOR_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False)

        logger.info(
            f"💾 Figma 인덱스 저장 완료: {self.settings.FIGMA_VECTOR_INDEX_PATH}"
        )

    # 디스크에 저장된 Figma 인덱스와 메타데이터를 메모리로 로드한다.
    def _load_index(self) -> None:
        if not (
            os.path.exists(self.settings.FIGMA_VECTOR_INDEX_PATH)
            and os.path.exists(self.settings.FIGMA_VECTOR_META_PATH)
        ):
            logger.info(
                "⚠️ Figma 인덱스/메타 파일이 없습니다. 새로 인덱싱하면서 생성됩니다."
            )
            return

        # FAISS 인덱스 로드
        index = faiss.read_index(self.settings.FIGMA_VECTOR_INDEX_PATH)

        # 메타데이터 로드
        with open(self.settings.FIGMA_VECTOR_META_PATH, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        self.index = index
        self.texts = meta_data.get("texts", [])
        self.metadatas = meta_data.get("metadatas", [])

        logger.info(f"✅ Figma 인덱스 로드 완료: {len(self.texts)} 개 청크")

    # Figma API를 호출해 전체 프로젝트 파일로부터 새 인덱스를 생성한다.
    async def _build_from_figma(self) -> any:
        """
        FIGMA_URL에 설정된 프로젝트/파일을 순회하면서
        FigmaFileLLMSummary를 만들고 인덱싱한다.
        """
        if not self.enabled:
            logger.info("Figma 비활성화 상태이므로 _build_from_figma를 스킵합니다.")
            return

        client = FigmaClient()

        # FIGMA_URL에서 team/project/file 정보 파싱
        data = parse_figma_team_project(self.settings.FIGMA_URL)
        logger.info(f"Figma URL 파싱 결과: {data}")

        project_id = data.get("project_id")
        if not project_id:
            logger.warning("FIGMA_URL에서 project_id를 찾을 수 없습니다. 빌드 스킵.")
            return False

        # 1) 프로젝트 파일 목록 조회
        try:
            project_files = await client.get_project_files(
                project_id, branch_data=False
            )
        except Exception:
            return False

        logger.info(f"Figma 프로젝트 내 파일 수: {len(project_files.files)}")

        all_entries: List[Dict[str, Any]] = []

        async def fetch_and_convert(file_meta):
            try:
                # 2) 파일 상세 조회
                detail = await client.get_file(file_meta.key)
                logger.info(f"Figma 파일 로드 완료: {detail.name} ({file_meta.key})")

                # 3) LLM 요약 스키마로 변환 (이미 네가 구현해 둔 메서드라고 가정)
                llm_summary: FigmaFileLLMSummary = client.get_file_llm_summary(detail)
                logger.info(f"Figma 파일 요약 생성 완료: {llm_summary.file_name}")

                # 4) 텍스트 청크로 변환
                return self._load_text_from_figma(
                    llm_summary,
                    source=file_meta.name,
                    chunk_mode=self.settings.FIGMA_CHUNK_MODE,
                )
            except Exception as e:
                logger.error(f"❌ Figma 파일 인덱싱 실패: {file_meta.name} - {e}")
                return []

        # 파일들 병렬 처리
        tasks = [fetch_and_convert(f) for f in project_files.files]
        chunk_lists = await asyncio.gather(*tasks)

        for chunk_list in chunk_lists:
            if not chunk_list:
                continue
            all_entries.extend(chunk_list)

        if not all_entries:
            logger.info("⚠️ Figma에서 인덱싱할 청크가 없습니다.")
            return False

        logger.info(f"📄 Figma에서 총 {len(all_entries)} 개 청크를 임베딩합니다...")
        self.add_entries(all_entries)

    # 필요 시 인덱스를 로드하거나 새로 빌드해서 Figma 벡터 스토어를 준비한다.
    async def ensure_vector_store(self):
        logger.info("Figma 벡터 스토어 초기화 확인...")

        # FIGMA_ENABLED = False 인 경우 인덱스 로드를 수행하지 않는다.
        if not self.enabled:
            logger.info("Figma 비활성화 상태이므로 벡터 스토어 로드를 생략합니다.")
            return

        if self.index is None:
            logger.info(
                f"Figma 벡터 스토어 로드 중: {self.settings.FIGMA_VECTOR_INDEX_PATH}"
            )

            # 인덱스/메타 파일이 둘 다 있으면 로드
            if os.path.exists(self.settings.FIGMA_VECTOR_INDEX_PATH) and os.path.exists(
                self.settings.FIGMA_VECTOR_META_PATH
            ):
                logger.info("📦 기존 Figma 인덱스를 로드합니다...")
                result = self._load_index()
            else:
                logger.info("Figma 인덱스가 없으므로 Figma API에서 새로 빌드합니다...")
                result = await self._build_from_figma()

            if result is None:
                logger.info("Figma 벡터 스토어 로드 완료")
            else:
                logger.info("Figma 벡터 스토어 로드 실패")
        else:
            logger.debug("✓ Figma 벡터 스토어 이미 초기화됨")

    # FigmaFileLLMSummary를 화면/섹션 단위의 텍스트 청크와 메타데이터로 변환한다.
    def _load_text_from_figma(
        self,
        figma_summary: FigmaFileLLMSummary,
        source: str = "figma",
        chunk_mode: str = "screen_only",  # "section_only", "screen_only", "both"
    ) -> List[Dict[str, Any]]:
        """
        chunk_mode 옵션:
        - "section_only": 섹션 단위 청크만 생성 (추천, 중복 없음, 검색 정확도 높음)
        - "screen_only": 화면 전체 청크만 생성 (청크 수 최소, 검색 정확도 약간 낮음)
        - "both": 둘 다 생성 (중복 있음, 이전 방식)
        """
        chunks: List[Dict[str, Any]] = []

        lm = figma_summary.last_modified
        if isinstance(lm, datetime):
            last_modified_str = lm.isoformat()
        else:
            last_modified_str = str(lm)

        for screen in figma_summary.screens:
            base_meta = {
                "source": source,
                "type": "figma_screen",
                "file_name": screen.file_name,
                "page_name": screen.page_name,
                "variant": screen.variant,
                "screen_id": screen.screen_id,
                "screen_path": screen.screen_path,
                "page_variant_key": f"{screen.page_name}:{screen.variant}",
                "last_modified": last_modified_str,
            }

            # 화면 단위 청크 (screen_only 또는 both 모드일 때)
            if chunk_mode in ("screen_only", "both"):
                lines: List[str] = []
                lines.append(f"# {screen.page_name} ({screen.variant})")
                lines.append("")

                for section in screen.sections:
                    lines.append(f"## {section.name}")
                    lines.append("")

                    if section.texts:
                        lines.append("### 텍스트")
                        for t in section.texts:
                            lines.append(f"- {t}")
                        lines.append("")

                    if section.controls:
                        lines.append("### 컨트롤")
                        for c in section.controls:
                            lines.append(f"- [{c.type.upper()}] {c.label}")
                        lines.append("")

                text = "\n".join(lines).strip()
                if text:
                    chunks.append(
                        {
                            "text": text,
                            "meta": {
                                **base_meta,
                                "content_type": "screen_summary",
                            },
                        }
                    )

            # 섹션 단위 청크 (section_only 또는 both 모드일 때)
            if chunk_mode in ("section_only", "both"):
                for section in screen.sections:
                    s_lines: List[str] = []
                    s_lines.append(
                        f"# {screen.page_name} > {section.name} ({screen.variant})"
                    )
                    s_lines.append("")

                    if section.texts:
                        s_lines.append("## 텍스트")
                        for t in section.texts:
                            s_lines.append(f"- {t}")
                        s_lines.append("")

                    if section.controls:
                        s_lines.append("## 컨트롤")
                        for c in section.controls:
                            s_lines.append(f"- [{c.type.upper()}] {c.label}")
                        s_lines.append("")

                    s_text = "\n".join(s_lines).strip()
                    if s_text:
                        chunks.append(
                            {
                                "text": s_text,
                                "meta": {
                                    **base_meta,
                                    "section_name": section.name,
                                    "section_id": section.node_id,
                                    "content_type": "section_detail",
                                },
                            }
                        )

        logger.info(f"📊 청크 생성 완료: {len(chunks)}개 (mode={chunk_mode})")
        return chunks

    # Figma에서 추출된 텍스트/메타데이터 엔트리를 인덱스에 추가한다.
    def add_entries(self, entries: List[Dict[str, Any]]) -> None:
        if not self.enabled:
            logger.info("Figma 비활성화 상태이므로 add_entries를 스킵합니다.")
            return

        if not entries:
            return

        texts = [e["text"] for e in entries]
        metas = [e["meta"] for e in entries]

        # 임베딩 생성 및 정규화
        embeddings = np.ascontiguousarray(
            embedding_service.embed_texts(texts), dtype="float32"
        )
        faiss.normalize_L2(embeddings)

        # 인덱스가 없으면 새로 만들고, 있으면 이어붙인다.
        if self.index is None:
            index = faiss.IndexFlatIP(self.dimension)
            index.add(embeddings)
            self.index = index
            self.texts = texts
            self.metadatas = metas
        else:
            self.index.add(embeddings)
            self.texts.extend(texts)
            self.metadatas.extend(metas)

        # 변경 사항 저장
        self._save_index()

    # 쿼리를 임베딩하여 Figma 인덱스에서 유사한 상위 청크들을 조회한다.
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("Figma RAG가 비활성화되어 검색할 수 없습니다.")

        if self.index is None:
            raise RuntimeError("Figma 벡터 인덱스가 초기화되지 않았습니다.")

        # 쿼리 임베딩
        query_vec = embedding_service.embed_query(query)
        faiss.normalize_L2(query_vec)

        # FAISS 검색
        scores, indices = self.index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append(
                {
                    "score": float(score),
                    "text": self.texts[idx],
                    "meta": self.metadatas[idx],
                }
            )

        return results


# 앱 전체에서 공유해서 쓸 Figma 전용 싱글톤 인스턴스
figma_rag_vector_store = FigmaRagVectorStore()
