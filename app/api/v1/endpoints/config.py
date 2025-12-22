from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_vector_db
from app.crud.config import get_app_config
from app.schemas.config import AppConfigResponse, AppConfigUpdate
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=AppConfigResponse, summary="현재 시스템 설정 조회")
def get_config(db: Session = Depends(get_vector_db)):
    """DB에 저장된 최신 시스템 설정을 가져옵니다."""
    config = get_app_config(db)
    if not config:
        raise HTTPException(status_code=404, detail="설정을 찾을 수 없습니다.")
    return config


@router.patch("/", response_model=AppConfigResponse, summary="시스템 설정 수정")
def update_config(obj_in: AppConfigUpdate, db: Session = Depends(get_vector_db)):
    """시스템 설정을 수정합니다. 전달된 필드만 업데이트됩니다."""
    config = get_app_config(db)
    if not config:
        raise HTTPException(status_code=404, detail="설정이 존재하지 않습니다.")

    update_data = obj_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(config, field, value)

    db.add(config)
    db.commit()
    db.refresh(config)

    logger.info(f"시스템 설정이 수정되었습니다: {update_data}")
    return config
