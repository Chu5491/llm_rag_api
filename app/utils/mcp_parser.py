import json
from typing import Optional, Dict, Any

def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    LLM이 반환한 문자열에서

    {
      "tool_name": "get_user",
      "arguments": { ... }
    }

    같은 JSON을 찾아 dict로 파싱.
    그 외에는 None 리턴.
    """
    if not isinstance(text, str):
        return None

    raw = text.strip()

    # ```json ... ``` 코드블록 제거
    if raw.startswith("```"):
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = raw[start : end + 1]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict) and "tool_name" in data and "arguments" in data:
        return data

    return None