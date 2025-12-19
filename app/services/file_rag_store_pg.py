# app/services/file_rag_store_pg.py
from typing import List, Dict, Any
import os
from pypdf import PdfReader

from app.core.config import get_settings
from app.services.embeddings import embedding_service
from app.core.logging import get_logger
from app.db.database import VectorSessionLocal
from app.models.vector_models import RagEmbedding

logger = get_logger(__name__)


class FileRagVectorStorePG:
    """
    Postgres + pgvector를 사용하는 파일 기반 RAG 스토어 클래스.
    PDF, TXT, MD 파일들의 내용을 임베딩하여 통합 테이블(RagEmbedding)에 저장하고 검색한다.
    """

    def __init__(self):
        self.settings = get_settings()
        self.dimension = embedding_service.dimension
        self.source_type = "file"  # 이 클래스에서 다루는 데이터의 소스 타입
        logger.info(
            f"FileRagVectorStorePG 초기화 완료 (source_type={self.source_type})"
        )

    def get_count(self) -> int:
        """현재 DB에 저장된 파일 소스의 총 청크 수를 반환한다."""
        with VectorSessionLocal() as db:
            return (
                db.query(RagEmbedding)
                .filter(RagEmbedding.source_type == self.source_type)
                .count()
            )

    async def ensure_vector_store(self):
        """
        벡터 저장소가 준비되었는지 확인한다.
        데이터가 하나도 없을 경우 uploads 디렉토리에 있는 파일들을 읽어 초기 빌드를 수행한다.
        """
        with VectorSessionLocal() as db:
            # 해당 소스 타입(file)의 데이터가 있는지 확인
            count = (
                db.query(RagEmbedding)
                .filter(RagEmbedding.source_type == self.source_type)
                .count()
            )
            if count == 0:
                logger.info(
                    f"DB에 '{self.source_type}' 데이터가 없으므로 '{self.settings.UPLOADS_DIR}' 폴더에서 초기 빌드를 시작합니다."
                )
                self._build_from_uploads()
            else:
                logger.info(
                    f"DB에 이미 '{self.source_type}' 데이터가 {count}개 존재합니다."
                )

    def _load_text_from_file(self, path: str) -> List[Dict[str, Any]]:
        """
        개별 파일(PDF, TXT, MD)에서 텍스트를 추출하여 청크 리스트로 반환한다.
        """
        results: List[Dict[str, Any]] = []
        ext = os.path.splitext(path)[1].lower()

        # 텍스트 및 마크다운 파일 처리
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
        # PDF 파일 처리 (페이지별로 청크 분리)
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
        """
        설정된 업로드 디렉토리를 순회하며 모든 지원 파일을 읽고,
        임베딩을 생성하여 DB의 RagEmbedding 테이블에 반영한다.
        """
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
            logger.info("인덱싱할 파일이 없습니다.")
            return

        logger.info(
            f"📄 총 {len(all_entries)} 개의 '{self.source_type}' 청크를 임베딩하여 DB에 저장합니다..."
        )

        # 텍스트 리스트를 한꺼번에 임베딩 서비스에 전달
        texts = [e["text"] for e in all_entries]
        embeddings = embedding_service.embed_texts(texts)

        # DB 세션을 열어 각 청크와 벡터를 저장
        with VectorSessionLocal() as db:
            for entry, emb in zip(all_entries, embeddings):
                new_emb = RagEmbedding(
                    source_type=self.source_type,
                    text=entry["text"],
                    embedding=emb.tolist(),
                    metadata_json=entry["meta"],
                )
                db.add(new_emb)
            db.commit()
        logger.info("✅ DB 저장 완료")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리 문자열을 임베딩하여 DB에서 유사도가 높은 상위 k개의 문서를 검색한다.
        source_type이 'file'인 항목만 필터링하여 검색한다.
        """
        # 쿼리를 벡터로 변환 (리스트 형태 [ [0.1, 0.2, ...] ])
        query_vec = embedding_service.embed_query(query)[0].tolist()

        with VectorSessionLocal() as db:
            # 코사인 거리(cosine_distance)를 기준으로 정렬하여 가장 가까운 문서 조회
            # (<=> operator in pgvector)
            results = (
                db.query(RagEmbedding)
                .filter(RagEmbedding.source_type == self.source_type)
                .order_by(RagEmbedding.embedding.cosine_distance(query_vec))
                .limit(top_k)
                .all()
            )

            output = []
            for res in results:
                output.append(
                    {
                        "score": 0.0,  # 필요한 경우 거리 값을 실제 점수로 환산하여 넣을 수 있음
                        "text": res.text,
                        "meta": res.metadata_json,
                    }
                )
            return output


# 외부 서비스에서 사용할 수 있도록 싱글톤 인스턴스 노출
file_rag_vector_store_pg = FileRagVectorStorePG()
