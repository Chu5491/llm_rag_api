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

@router.post("/cancel/{history_id}")
async def cancel_generation(history_id: int, db: Session = Depends(get_vector_db)):
    """
    실행 중인 RAG 생성 작업을 취소합니다.
    """
    # DB에서 히스토리 확인
    history = get_history_by_id(db, history_id)
    if not history:
        raise HTTPException(status_code=404, detail="해당 이력을 찾을 수 없습니다.")

    # 이미 완료된 작업인지 확인
    if history.status in ["success", "failed", "cancelled"]:
        return {
            "status": "already_completed",
            "detail": "이미 완료되거나 취소된 작업입니다.",
        }

    # 활성 작업에서 찾아서 취소
    task = get_active_tasks().get(history_id)
    if task and not task.done():
        task.cancel()
        try:
            await task  # 취소 완료 대기
        except asyncio.CancelledError:
            pass

    # DB 상태 업데이트
    complete_history(
        db, history_id, status="cancelled", summary="사용자 요청에 의해 취소됨"
    )

    return {"status": "cancelled", "detail": "작업이 성공적으로 취소되었습니다."}