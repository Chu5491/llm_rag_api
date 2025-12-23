from sqlalchemy.orm import Session
from app.models.history import GenerateHistory
from datetime import datetime, timezone, timedelta
from typing import Any, List

# KST 타임존 설정 (UTC+9)
KST = timezone(timedelta(hours=9))


def create_history(
    db: Session, title: str, source_type: str, total_batches: int, model_name: str
):
    """실행 이력을 처음 생성 (running 상태)"""
    db_history = GenerateHistory(
        title=title,
        source_type=source_type,
        total_batches=total_batches,
        current_batch=0,
        progress=0,
        model_name=model_name,
        status="running",
        logs=f"[{datetime.now(KST).strftime('%H:%M:%S')}] 작업 시작\n",
    )
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history


def update_progress(
    db: Session, history_id: int, current_batch: int, log_msg: str = None
):
    """실시간 진행률 및 로그 업데이트"""
    history = db.query(GenerateHistory).filter(GenerateHistory.id == history_id).first()
    if history:
        history.current_batch = current_batch
        if history.total_batches > 0:
            history.progress = int((current_batch / history.total_batches) * 100)

        if log_msg:
            timestamp = datetime.now(KST).strftime("%H:%M:%S")
            history.logs = (history.logs or "") + f"[{timestamp}] {log_msg}\n"

        db.commit()
        db.refresh(history)
    return history


def complete_history(
    db: Session, history_id: int, status: str, summary: str, result_data: Any = None
):
    """작업 종료 처리 (success/failed)"""
    history = db.query(GenerateHistory).filter(GenerateHistory.id == history_id).first()
    if history:
        history.status = status
        history.summary = summary
        history.result_data = result_data
        history.finished_at = datetime.now(timezone.utc)
        history.duration = history.finished_at - history.started_at
        if status == "success":
            history.progress = 100

        timestamp = datetime.now(KST).strftime("%H:%M:%S")
        history.logs = (history.logs or "") + f"[{timestamp}] 작업 {status}\n"

        db.commit()
        db.refresh(history)
    return history


def get_histories(db: Session, skip: int = 0, limit: int = 20) -> List[GenerateHistory]:
    """전체 히스토리 목록 조회 (최신순)"""
    return (
        db.query(GenerateHistory)
        .order_by(GenerateHistory.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_history_by_id(db: Session, history_id: int) -> GenerateHistory:
    """히스토리 단건 조회 (상세 로그 및 결과 확인용)"""
    return db.query(GenerateHistory).filter(GenerateHistory.id == history_id).first()
