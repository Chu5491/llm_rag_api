from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_vector_db
from app.crud.history import get_histories, get_history_by_id
from app.schemas.history import HistoryResponse, HistoryDetailResponse

router = APIRouter(prefix="/history", tags=["History"])


@router.get("", response_model=List[HistoryResponse])
def histories(skip: int = 0, limit: int = 20, db: Session = Depends(get_vector_db)):
    """
    테스트케이스 생성 이력 목록을 조회합니다. (최신순)
    """
    histories = get_histories(db, skip=skip, limit=limit)
    return histories


@router.get("/{history_id}", response_model=HistoryDetailResponse)
def history_detail(history_id: int, db: Session = Depends(get_vector_db)):
    """
    특정 이력의 상세 정보(로그, 결과 데이터 등)를 조회합니다.
    실시간 진행률 확인(Polling) 등에 사용됩니다.
    """
    history = get_history_by_id(db, history_id=history_id)
    if history is None:
        raise HTTPException(status_code=404, detail="해당 이력을 찾을 수 없습니다.")
    return history
