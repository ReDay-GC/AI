"""
AI 기억 생성 로직
현재: Anthropic Claude API 사용 (claude-sonnet-4-5)
"""

import json
import re
import os
import base64
import io
import anthropic
from PIL import Image
from prompt_builder import build_memory_prompt

# ── 설정 ──────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-5"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
# ────────────────────────────────────────────────────────────


def generate_memory_with_ai(records: list, photo_paths: list = []) -> dict:
    """
    records: [{"type": str, "content": str, "time": str, "location": str}, ...]
    photo_paths: 사진 파일 절대 경로 목록
    returns: {"title": str, "summary": str, "tags": list[str], "people": list[str]}
    """
    prompt = build_memory_prompt(records)

    # 메시지 content 구성
    content = []

    # 사진이 있으면 Vision으로 분석 (5MB 초과 시 리사이즈)
    for b64_data in photo_paths:
        try:
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # 5MB 이하가 될 때까지 축소
            max_size = 4 * 1024 * 1024  # 4MB 여유있게
            quality = 85
            scale = 1.0
            while True:
                buf = io.BytesIO()
                w = int(img.width * scale)
                h = int(img.height * scale)
                img.resize((w, h), Image.LANCZOS).save(buf, format="JPEG", quality=quality)
                if buf.tell() <= max_size or scale < 0.2:
                    break
                scale *= 0.8

            resized_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": resized_b64}
            })
        except Exception:
            pass

    content.append({"type": "text", "text": prompt})

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}]
        )
        raw = response.content[0].text
        return _parse_response(raw, records)

    except Exception as e:
        raise RuntimeError(f"AI 호출 실패: {e}")


_PEOPLE_PLACEHOLDERS = {
    "그룹명", "이름", "누군가", "친구", "사람", "인물", "누구",
    "name", "person", "someone", "group"
}

def _filter_placeholder_people(people: list) -> list:
    result = []
    for p in people:
        if not p:
            continue
        if len(p) <= 1:
            continue
        if p in _PEOPLE_PLACEHOLDERS:
            continue
        result.append(p)
    return result


def _parse_response(raw: str, records: list) -> dict:
    """JSON 파싱 실패 시 rule-based 폴백"""
    try:
        # JSON 블록 추출 시도
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            title = str(data.get("title", "")).strip()
            summary = str(data.get("summary", "")).strip()
            tags = data.get("tags", [])
            if isinstance(tags, list):
                tags = [str(t).strip() for t in tags]
            else:
                tags = []

            people = data.get("people", [])
            if isinstance(people, list):
                people = [str(p).strip() for p in people]
                people = _filter_placeholder_people(people)
            else:
                people = []

            if title and summary:
                return {"title": title, "summary": summary, "tags": tags, "people": people}
    except (json.JSONDecodeError, KeyError):
        pass

    # 폴백: rule-based
    return _rule_based_fallback(records)


# ── Rule-based 폴백 (AI 실패 시) ─────────────────────────────
def _rule_based_fallback(records: list) -> dict:
    date = records[0].get("time", "")[:10] if records else ""
    location = records[0].get("location", "") if records else ""

    try:
        parts = date.split("-")
        title = f"{int(parts[1])}월 {int(parts[2])}일의 기억"
    except Exception:
        title = "하루의 기억"

    contents = [r.get("content", "") for r in records if r.get("content")]
    summary = " ".join(contents[:2]) if contents else "기록이 없습니다."

    tags = []
    if location:
        tags.append(location.split()[-1] if " " in location else location)
    tags += ["기억", "하루"]

    return {"title": title, "summary": summary[:100], "tags": tags[:5], "people": []}
