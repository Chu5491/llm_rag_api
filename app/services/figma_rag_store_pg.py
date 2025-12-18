# app/services/figma_rag_store_pg.py
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from app.core.config import get_settings
from app.services.embeddings import embedding_service
from app.core.logging import get_logger
from app.db.database import SessionLocal
from app.models.vector_models import FigmaEmbedding
from app.schemas.figma import FigmaFileLLMSummary
from app.services.figma_client import FigmaClient
from app.utils.figma_url_parser import parse_figma_team_project

logger = get_logger(__name__)


class FigmaRagVectorStorePG:
    """Postgres + pgvector를 사용하는 Figma RAG 스토어"""

    def __init__(self):
        self.settings = get_settings()
        self.enabled = bool(self.settings.FIGMA_ENABLED)
        logger.info("FigmaRagVectorStorePG 초기화 완료")

    async def ensure_vector_store(self):
        if not self.enabled:
            return

        with SessionLocal() as db:
            count = db.query(FigmaEmbedding).count()
            if count == 0:
                logger.info("Figma DB에 데이터가 없으므로 API에서 빌드합니다.")
                await self._build_from_figma()
            else:
                logger.info(f"Figma DB에 이미 {count}개의 데이터가 존재합니다.")

    async def _build_from_figma(self):
        if not self.enabled:
            return

        client = FigmaClient()
        data = parse_figma_team_project(self.settings.FIGMA_URL)
        project_id = data.get("project_id")
        if not project_id:
            return

        try:
            project_files = await client.get_project_files(project_id)
        except Exception:
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

        tasks = [fetch_and_convert(f) for f in project_files.files]
        chunk_lists = await asyncio.gather(*tasks)
        for cl in chunk_lists:
            all_entries.extend(cl)

        if all_entries:
            self.add_entries(all_entries)

    def _load_text_from_figma(
        self, figma_summary: FigmaFileLLMSummary, source: str, chunk_mode: str
    ) -> List[Dict[str, Any]]:
        # 기존 figma_rag_store.py의 로직과 동일 (생략 없이 구현)
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
        if not entries:
            return
        texts = [e["text"] for e in entries]
        embeddings = embedding_service.embed_texts(texts)

        with SessionLocal() as db:
            for entry, emb in zip(entries, embeddings):
                db.add(
                    FigmaEmbedding(
                        text=entry["text"],
                        embedding=emb.tolist(),
                        metadata_json=entry["meta"],
                    )
                )
            db.commit()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        query_vec = embedding_service.embed_query(query)[0].tolist()
        with SessionLocal() as db:
            results = (
                db.query(FigmaEmbedding)
                .order_by(FigmaEmbedding.embedding.cosine_distance(query_vec))
                .limit(top_k)
                .all()
            )
            return [
                {"score": 0.0, "text": r.text, "meta": r.metadata_json} for r in results
            ]


figma_rag_vector_store_pg = FigmaRagVectorStorePG()
