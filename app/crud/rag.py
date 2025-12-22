from typing import List
from sqlalchemy.orm import Session
from app.models.rag_embeddings import RagEmbedding


def get_rag_count_by_source(db: Session, source_type: str) -> int:
    """특정 소스 타입의 청크 개수를 조회합니다."""
    return (
        db.query(RagEmbedding).filter(RagEmbedding.source_type == source_type).count()
    )


def create_rag_embeddings(db: Session, embeddings_data: List[dict]):
    """여러 개의 임베딩 데이터를 한꺼번에 저장합니다."""
    objects = [
        RagEmbedding(
            source_type=item["source_type"],
            text=item["text"],
            embedding=item["embedding"],
            metadata_json=item.get("metadata_json"),
        )
        for item in embeddings_data
    ]
    db.bulk_save_objects(objects)
    db.commit()


def search_similar_embeddings(
    db: Session, query_vec: List[float], source_type: str, top_k: int = 5
) -> List[RagEmbedding]:
    """단일 소스 타입 유사도 검색"""
    return (
        db.query(RagEmbedding)
        .filter(RagEmbedding.source_type == source_type)
        .order_by(RagEmbedding.embedding.cosine_distance(query_vec))
        .limit(top_k)
        .all()
    )


def search_all_sources(
    db: Session, query_vec: List[float], top_k: int = 10
) -> List[RagEmbedding]:
    """파일과 피그마를 포함한 모든 소스에서 유사도 검색"""
    return (
        db.query(RagEmbedding)
        .order_by(RagEmbedding.embedding.cosine_distance(query_vec))
        .limit(top_k)
        .all()
    )
