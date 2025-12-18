# utils/mcp_parser.py
import json
import re
from typing import Optional, Dict, Any
from app.core.logging import get_logger

logger = get_logger(__name__)


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    LLM 응답에서 JSON 형식의 툴 호출 파싱
    - 다양한 포맷 지원 (JSON, 마크다운 블록, 일반 텍스트)
    - 부분 복구 시도
    """
    if not isinstance(text, str):
        return None

    raw = text.strip()

    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        # ```json { ... } ``` 형태
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

    # JSON 객체 추출 시도
    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        logger.warning(f"[MCP Parser] JSON 구조를 찾을 수 없음: {raw[:100]}")
        return None

    candidate = raw[start : end + 1]

    # JSON 파싱 with 에러 핸들링
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.error(f"[MCP Parser] JSON 파싱 실패: {e}\nCandidate: {candidate[:200]}")
        # 부분 복구 시도: 따옴표 이스케이프 문제 해결
        try:
            # 줄바꿈 문자 이스케이프
            candidate_fixed = candidate.replace("\n", "\\n")
            data = json.loads(candidate_fixed)
            logger.info("[MCP Parser] 부분 복구 성공 (줄바꿈 이스케이프)")
        except Exception as e:
            logger.error(f"[MCP Parser] 부분 복구 실패: {e}")
            return None

    # 스키마 검증
    if not isinstance(data, dict):
        logger.error(f"[MCP Parser] 최상위가 dict가 아님: {type(data)}")
        return None

    if "tool_name" not in data or "arguments" not in data:
        logger.warning(f"[MCP Parser] 필수 필드 누락: {data.keys()}")
        return None

    if not isinstance(data.get("arguments"), dict):
        logger.warning(
            f"[MCP Parser] arguments가 dict가 아님: {type(data.get('arguments'))}"
        )
        return None

    return data
