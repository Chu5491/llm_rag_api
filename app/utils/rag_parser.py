# app/utils/rag_parser.py
import json
import re
from json.decoder import JSONDecodeError
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


def _strip_code_fence(text: str) -> str:
    """```json, ``` 같은 코드블록 마커를 제거한다."""
    if not text:
        return text

    t = text.strip()

    # 앞쪽 ```json / ``` 제거
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    # 뒤쪽 ``` 제거
    t = re.sub(r"```$", "", t).strip()
    return t


def _auto_close_brackets(s: str) -> str:
    """
    LLM이 마지막에 }나 ]를 빼먹은 경우를 대비해서
    여는/닫는 괄호 개수를 맞춰준다.
    """
    open_curly = s.count("{")
    close_curly = s.count("}")
    open_square = s.count("[")
    close_square = s.count("]")

    if close_curly < open_curly:
        s = s + ("}" * (open_curly - close_curly))
    if close_square < open_square:
        s = s + ("]" * (open_square - close_square))

    return s


def _remove_trailing_commas(s: str) -> str:
    """
    객체/배열 끝에 붙은 trailing comma 를 제거한다.
    예: {"a": 1,} → {"a": 1}
        [ {...}, ] → [ {...} ]
    """
    return re.sub(r",(\s*[\]}])", r"\1", s)


def _find_array_candidates(text: str) -> List[str]:
    """
    텍스트 안에서 [ ... ] 구간들을 non-greedy로 모두 찾아서 후보 리스트로 반환한다.
    """
    candidates: List[str] = []
    for m in re.finditer(r"\[[\s\S]*?\]", text):
        candidates.append(m.group(0).strip())
    return candidates


def extract_json_array(text: str) -> str:
    """
    LLM이 출력한 텍스트에서 JSON 배열 문자열을 최대한 뽑아낸다.

    - 반환 타입은 여전히 str (call site: json.loads(extract_json_array(...)))
    - 이미 전체가 JSON 배열이거나, "배열을 문자열로 감싼" 형태도 처리한다.
    """
    if not text:
        msg = "빈 LLM 응답입니다."
        logger.error("[RAG-PARSER] %s", msg)
        raise JSONDecodeError(msg, text, 0)

    # 0) 코드블록 마커 제거
    cleaned = _strip_code_fence(text).strip()

    # 0-1) 전체를 한 번 JSON으로 파싱 시도
    #   - case A: cleaned == '[{...}]'  → 바로 배열
    #   - case B: cleaned == '"[{\"...\"}]"' → 문자열 안에 배열 JSON
    try:
        top = json.loads(cleaned)

        # A. 최상위가 이미 리스트인 경우 → 그냥 예쁘게 다시 덤프해서 반환
        if isinstance(top, list):
            logger.debug(
                "[RAG-PARSER] 최상위 JSON이 이미 배열, 그대로 사용 (len=%d)",
                len(top),
            )
            return json.dumps(top, ensure_ascii=False)

        # B. 최상위가 문자열인데, 그 안이 또 JSON 배열일 수 있는 경우
        if isinstance(top, str):
            inner = top.strip()
            try:
                inner_obj = json.loads(inner)
                if isinstance(inner_obj, list):
                    logger.debug(
                        "[RAG-PARSER] 최상위가 문자열이지만, 내부가 JSON 배열이라서 내부 배열 사용 (len=%d)",
                        len(inner_obj),
                    )
                    return json.dumps(inner_obj, ensure_ascii=False)
            except JSONDecodeError:
                # 내부가 JSON이 아니면 그냥 패스하고 기존 로직으로 진행
                pass
    except JSONDecodeError:
        # 전체가 JSON이 아닐 수도 있으니, 이 경우는 무시하고 아래 후보 탐색으로 진행
        pass

    # === 여기부터는 기존 후보/보정 로직 그대로 ===

    candidates = _find_array_candidates(cleaned)
    if not candidates:
        candidates = [cleaned.strip()]

    last_error: Optional[Exception] = None

    for cand_idx, cand in enumerate(candidates, start=1):
        for attempt in range(3):
            if attempt == 0:
                s = cand
                desc = "raw"
            elif attempt == 1:
                s = _remove_trailing_commas(cand)
                desc = "no_trailing_commas"
            else:
                s = _auto_close_brackets(_remove_trailing_commas(cand))
                desc = "auto_closed"

            try:
                obj = json.loads(s)
            except JSONDecodeError as e:
                last_error = e
                logger.debug(
                    "[RAG-PARSER] 후보 %d / 시도 %s JSONDecodeError: %s",
                    cand_idx,
                    desc,
                    e,
                )
                continue

            if isinstance(obj, list):
                logger.debug(
                    "[RAG-PARSER] 후보 %d / 시도 %s 에서 JSON 배열 파싱 성공 (len=%d)",
                    cand_idx,
                    desc,
                    len(obj),
                )
                return s

            last_error = ValueError(f"JSON 루트 타입이 list가 아님: {type(obj)}")
            logger.debug(
                "[RAG-PARSER] 후보 %d / 시도 %s 루트 타입이 list가 아님: %s",
                cand_idx,
                desc,
                type(obj),
            )
            break

    snippet = cleaned[:300]

    if isinstance(last_error, JSONDecodeError):
        err = last_error
    else:
        msg = "LLM 응답에서 유효한 JSON 배열을 찾지 못했습니다."
        if last_error:
            msg += f" 마지막 오류: {last_error}"
        err = JSONDecodeError(msg, cleaned, 0)

    logger.error(
        "[RAG-PARSER] JSON 배열 파싱 실패: %s / raw_snippet=%s",
        err,
        snippet,
    )
    raise err
