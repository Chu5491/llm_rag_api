# app/services/file_rag_store_pg.py
from typing import List, Dict, Any
import os
from pypdf import PdfReader
from sqlalchemy import text

from app.core.config import get_settings
from app.services.embeddings import embedding_service
from app.core.logging import get_logger
from app.db.database import SessionLocal, engine, Base
from app.models.vector_models import FileEmbedding

# pgvector 확장 및 테이블 생성을 위해 실행
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
Base.metadata.create_all(bind=engine)

logger = get_logger(__name__)


class FileRagVectorStorePG:
    """Postgres + pgvector를 사용하는 파일 RAG 스토어"""

    def __init__(self):
        self.settings = get_settings()
        self.dimension = embedding_service.dimension
        logger.info("FileRagVectorStorePG 초기화 완료")

    async def ensure_vector_store(self):
        """기존 FAISS 버전과 인터페이스를 맞추기 위한 메서드.
        DB의 경우 이미 테이블이 생성되어 있으므로, 필요 시 초기화 로직 수행"""
        with SessionLocal() as db:
            count = db.query(FileEmbedding).count()
            if count == 0:
                logger.info(
                    "DB에 데이터가 없으므로 uploads 폴더에서 초기 빌드를 시작합니다."
                )
                self._build_from_uploads()
            else:
                logger.info(f"DB에 이미 {count}개의 데이터가 존재합니다.")

    def _load_text_from_file(self, path: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        ext = os.path.splitext(path)[1].lower()

        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                results.append(
                    {
                        "text": content,
                        "meta": {"source": os.path.basename(path), "type": "text"},
                    }
                )
        elif ext == ".pdf":
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages):
                text_content = page.extract_text() or ""
                if text_content.strip():
                    results.append(
                        {
                            "text": text_content,
                            "meta": {
                                "source": os.path.basename(path),
                                "type": "pdf",
                                "page": page_num,
                            },
                        }
                    )
        return results

    def _build_from_uploads(self) -> None:
        uploads_dir = self.settings.UPLOADS_DIR
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir, exist_ok=True)
            return

        all_entries = []
        for filename in os.listdir(uploads_dir):
            path = os.path.join(uploads_dir, filename)
            if os.path.isfile(path):
                entries = self._load_text_from_file(path)
                all_entries.extend(entries)

        if not all_entries:
            return

        logger.info(
            f"📄 총 {len(all_entries)} 개의 청크를 임베딩하여 DB에 저장합니다..."
        )

        texts = [e["text"] for e in all_entries]
        embeddings = embedding_service.embed_texts(texts)

        with SessionLocal() as db:
            for entry, emb in zip(all_entries, embeddings):
                new_emb = FileEmbedding(
                    text=entry["text"],
                    embedding=emb.tolist(),
                    metadata_json=entry["meta"],
                )
                db.add(new_emb)
            db.commit()
        logger.info("✅ DB 저장 완료")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = embedding_service.embed_query(query)[0].tolist()

        with SessionLocal() as db:
            # l2_distance (<->), cosine_distance (<=>), inner_product (<#>)
            # 여기서는 FAISS FlatIP와 유사하게 inner_product 또는 cosine_distance 사용 가능
            # nomic-embed-text는 cosine 유사도가 일반적임
            results = (
                db.query(FileEmbedding)
                .order_by(FileEmbedding.embedding.cosine_distance(query_vec))
                .limit(top_k)
                .all()
            )

            output = []
            for res in results:
                output.append(
                    {
                        "score": 0.0,  # 필요 시 거리 계산 결과 추가 가능
                        "text": res.text,
                        "meta": res.metadata_json,
                    }
                )
            return output


file_rag_vector_store_pg = FileRagVectorStorePG()
