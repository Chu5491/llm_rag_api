# app/services/figma_rag_store_pg.py
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from app.core.config import get_settings
from app.services.embeddings import embedding_service
from app.core.logging import get_logger
from app.db.database import SessionLocal
from app.models.vector_models import RagEmbedding
from app.schemas.figma import FigmaFileLLMSummary
from app.services.figma_client import FigmaClient
from app.utils.figma_url_parser import parse_figma_team_project

logger = get_logger(__name__)


class FigmaRagVectorStorePG:
    """
    Postgres + pgvector를 사용하는 Figma 전용 RAG 스토어 클래스.
    Figma API에서 추출한 화면/섹션 정보를 통합 테이블(RagEmbedding)에 저장한다.
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = bool(self.settings.FIGMA_ENABLED)
        self.source_type = "figma"  # 소스 타입 구분용
        logger.info(
            f"FigmaRagVectorStorePG 초기화 완료 (source_type={self.source_type})"
        )

    def get_count(self) -> int:
        """현재 DB에 저장된 Figma 소스의 총 청크 수를 반환한다."""
        with SessionLocal() as db:
            return (
                db.query(RagEmbedding)
                .filter(RagEmbedding.source_type == self.source_type)
                .count()
            )

    async def ensure_vector_store(self):
        """
        Figma 벡터 저장소가 준비되었는지 확인한다. 비어있을 경우 Figma API에서 데이터를 가져온다.
        """
        if not self.enabled:
            return

        with SessionLocal() as db:
            # figma 타입의 데이터가 있는지 확인
            count = (
                db.query(RagEmbedding)
                .filter(RagEmbedding.source_type == self.source_type)
                .count()
            )
            if count == 0:
                logger.info(
                    f"DB에 '{self.source_type}' 데이터가 없으므로 API에서 빌드합니다."
                )
                await self._build_from_figma()
            else:
                logger.info(
                    f"DB에 이미 '{self.source_type}' 데이터가 {count}개 존재합니다."
                )

    async def _build_from_figma(self):
        """
        Figma API를 통해 프로젝트 파일들을 조회하고, LLM 요약 정보를 생성하여 DB에 저장한다.
        """
        if not self.enabled:
            return

        client = FigmaClient()
        data = parse_figma_team_project(self.settings.FIGMA_URL)
        project_id = data.get("project_id")
        if not project_id:
            logger.warning("Figma URL에서 project_id를 찾을 수 없습니다.")
            return

        try:
            project_files = await client.get_project_files(project_id)
        except Exception as e:
            logger.error(f"Figma 프로젝트 파일 목록 조회 실패: {e}")
            return

        all_entries: List[Dict[str, Any]] = []

        async def fetch_and_convert(file_meta):
            try:
                detail = await client.get_file(file_meta.key)
                llm_summary = client.get_file_llm_summary(detail)
                return self._load_text_from_figma(
                    llm_summary,
                    source=file_meta.name,
                    chunk_mode=self.settings.FIGMA_CHUNK_MODE,
                )
            except Exception as e:
                logger.error(f"Figma 파일 인덱싱 실패: {file_meta.name} - {e}")
                return []

        # 각 파일별로 병렬 페치 및 변환 수행
        tasks = [fetch_and_convert(f) for f in project_files.files]
        chunk_lists = await asyncio.gather(*tasks)
        for cl in chunk_lists:
            all_entries.extend(cl)

        if all_entries:
            self.add_entries(all_entries)

    def _load_text_from_figma(
        self, figma_summary: FigmaFileLLMSummary, source: str, chunk_mode: str
    ) -> List[Dict[str, Any]]:
        """
        FigmaFileLLMSummary 객체를 검색 가능한 텍스트 청크 단위로 파싱한다.
        """
        chunks = []
        lm = figma_summary.last_modified
        last_modified_str = lm.isoformat() if isinstance(lm, datetime) else str(lm)

        for screen in figma_summary.screens:
            base_meta = {
                "source": source,
                "type": "figma_screen",
                "file_name": screen.file_name,
                "page_name": screen.page_name,
                "variant": screen.variant,
                "screen_id": screen.screen_id,
                "screen_path": screen.screen_path,
                "last_modified": last_modified_str,
            }

            # 화면 요약 단위 청크
            if chunk_mode in ("screen_only", "both"):
                lines = [f"# {screen.page_name} ({screen.variant})", ""]
                for section in screen.sections:
                    lines.append(f"## {section.name}\n")
                    if section.texts:
                        lines.append("### 텍스트")
                        for t in section.texts:
                            lines.append(f"- {t}")
                    if section.controls:
                        lines.append("\n### 컨트롤")
                        for c in section.controls:
                            lines.append(f"- [{c.type.upper()}] {c.label}")
                text = "\n".join(lines).strip()
                if text:
                    chunks.append(
                        {
                            "text": text,
                            "meta": {**base_meta, "content_type": "screen_summary"},
                        }
                    )

            # 섹션 상세 단위 청크
            if chunk_mode in ("section_only", "both"):
                for section in screen.sections:
                    s_lines = [
                        f"# {screen.page_name} > {section.name} ({screen.variant})",
                        "",
                    ]
                    if section.texts:
                        s_lines.append("## 텍스트")
                        for t in section.texts:
                            s_lines.append(f"- {t}")
                    if section.controls:
                        s_lines.append("\n## 컨트롤")
                        for c in section.controls:
                            s_lines.append(f"- [{c.type.upper()}] {c.label}")
                    s_text = "\n".join(s_lines).strip()
                    if s_text:
                        chunks.append(
                            {
                                "text": s_text,
                                "meta": {
                                    **base_meta,
                                    "section_name": section.name,
                                    "content_type": "section_detail",
                                },
                            }
                        )
        return chunks

    def add_entries(self, entries: List[Dict[str, Any]]):
        """
        생성된 청크들을 임베딩하여 DB에 저장한다.
        """
        if not entries:
            return
        texts = [e["text"] for e in entries]
        embeddings = embedding_service.embed_texts(texts)

        with SessionLocal() as db:
            for entry, emb in zip(entries, embeddings):
                db.add(
                    RagEmbedding(
                        source_type=self.source_type,
                        text=entry["text"],
                        embedding=emb.tolist(),
                        metadata_json=entry["meta"],
                    )
                )
            db.commit()
        logger.info(f"✅ Figma 데이터를 {len(entries)}개 저장했습니다.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Figma 소스 데이터 중에서 유사도가 높은 상위 k개를 검색한다.
        """
        if not self.enabled:
            return []

        query_vec = embedding_service.embed_query(query)[0].tolist()

        with SessionLocal() as db:
            results = (
                db.query(RagEmbedding)
                .filter(RagEmbedding.source_type == self.source_type)
                .order_by(RagEmbedding.embedding.cosine_distance(query_vec))
                .limit(top_k)
                .all()
            )
            return [
                {"score": 0.0, "text": r.text, "meta": r.metadata_json} for r in results
            ]


# 싱글톤 인스턴스
figma_rag_vector_store_pg = FigmaRagVectorStorePG()
