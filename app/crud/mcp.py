from typing import Optional, List, Dict, Any
from app.db.database import get_mcp_db


def _execute_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
    """
    SQL 쿼리를 실행하고 결과를 딕셔너리 리스트로 반환합니다.

    Args:
        query: 실행할 SQL 쿼리 문자열
        params: 쿼리 파라미터 (튜플)
    Returns:
        쿼리 결과를 담은 딕셔너리 리스트
    """
    conn = get_mcp_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            if cursor.description:  # 결과가 있는 경우에만 처리
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
    finally:
        conn.close()


def _execute_query_one(query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
    """
    단일 행을 조회하는 SQL 쿼리를 실행하고 결과를 딕셔너리로 반환합니다.
    결과가 없을 경우 None을 반환합니다.
    """
    results = _execute_query(query, params)
    return results[0] if results else None


def fetch_all_models_sync(only_used: bool = False) -> List[Dict[str, Any]]:
    """
    AI 모델 목록을 조회합니다.

    Args:
        only_used: True인 경우 is_used가 true인 모델만 필터링
    Returns:
        AI 모델 정보 리스트
    """
    query = "SELECT * FROM ai_model"
    if only_used:
        query += " WHERE is_used = true"
    query += " ORDER BY id"
    return _execute_query(query)


def fetch_model_by_name_sync(model_name: str) -> Optional[Dict[str, Any]]:
    """
    모델 이름으로 특정 AI 모델을 조회합니다.

    Args:
        model_name: 조회할 모델 이름
    Returns:
        모델 정보 딕셔너리 (없을 경우 None)
    """
    query = "SELECT * FROM ai_model WHERE model_name = %s"
    return _execute_query_one(query, (model_name,))


def fetch_eval_session_stats_sync() -> Dict[str, Any]:
    """
    평가 세션의 상태별 통계를 조회합니다.

    Returns:
        {
            "running": 실행 중인 세션 수,
            "done": 완료된 세션 수,
            "error": 오류 발생한 세션 수,
            "raw": 모든 상태별 원본 데이터
        }
    """
    query = """
        SELECT
            eval_status,
            COUNT(*) as count
        FROM eval_session_plan
        GROUP BY eval_status
    """
    results = _execute_query(query)

    # 상태별 기본값 초기화
    counts = {
        "RUNNING": 0,
        "DONE": 0,
        "ERROR": 0,
    }
    raw = []

    # 결과 처리
    for result in results:
        status = result["eval_status"]
        count = result["count"]
        status_upper = (status or "").upper()

        # 원본 데이터 저장
        raw.append(
            {
                "eval_status": status,
                "count": count,
            }
        )

        # 주요 상태별로 카운트
        if status_upper in counts:
            counts[status_upper] = count

    return {
        "running": counts["RUNNING"],
        "done": counts["DONE"],
        "error": counts["ERROR"],
        "raw": raw,
    }
