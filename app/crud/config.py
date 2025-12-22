# app/crud/config.py 수정 버전
from sqlalchemy.orm import Session
from app.models.configuration import Configuration


def get_app_config(db: Session):
    """DB에서 최신 설정을 가져옵니다."""
    return db.query(Configuration).order_by(Configuration.id).first()
