"""
[임시 로컬 서버] FastAPI + Claude API
나중에 실서버로 옮길 때 BASE_URL, 인증, DB 연결만 교체하면 됨
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
from collections import Counter
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import httpx
from memory_generator import (
    generate_memory_with_ai,
    generate_insight_with_ai,
    parse_search_query,
    generate_daily_comment,
    generate_embedding,
    analyze_record_text,
    extract_record_keywords,
    classify_memory_tags,
    classify_activity_type,
    extract_search_keywords,
    analyze_location_patterns,
)
from openai import OpenAI

SPRING_BOOT_BASE_URL = os.environ.get("SPRING_BOOT_BASE_URL", "http://15.164.99.114:8080")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CATEGORY_KEYWORDS = {
    "놀이공원": ["놀이공원", "에버랜드", "롯데월드", "테마파크", "롤러코스터", "놀이기구", "퍼레이드"],
    "카페":    ["카페", "스타벅스", "투썸", "이디야", "커피"],
    "영화관":  ["영화관", "CGV", "메가박스", "롯데시네마", "영화"],
    "바다":    ["바다", "해변", "해운대", "광안리", "강릉", "제주"],
    "운동":    ["운동", "헬스", "헬스장", "러닝", "요가", "필라테스", "달리기", "마라톤", "완주", "조깅"],
    "공부":    ["공부", "학습", "시험", "과제", "도서관"],
    "식사":    ["밥", "식사", "음식", "맛집", "식당", "맛"],
}

SAD_WORDS     = ["슬픔", "슬픈", "힘듦", "힘든", "힘들", "지침", "지친", "피곤", "속상", "눈물"]
HAPPY_WORDS   = ["행복", "즐거", "기쁨", "기쁜", "좋았"]
EXCITED_WORDS = ["신남", "신나", "설레", "흥미", "기대"]
ANGRY_WORDS   = ["화남", "화난", "화나", "짜증","분노", "분노한", "분노하다", "빡침", "빡치", "개화", "억울", "열받"]
EMOTION_GROUPS = [SAD_WORDS, HAPPY_WORDS, EXCITED_WORDS, ANGRY_WORDS]


def keyword_bonus(query: str, memory_text: str) -> float:
    q = query.lower()
    m = memory_text.lower()

    bonus = 0.0

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            if any(kw in m for kw in keywords):
                bonus += 0.25
                break

    for group in EMOTION_GROUPS:
        if any(w in q for w in group):
            if any(w in m for w in group):
                bonus += 0.2
                break

    return bonus


app = FastAPI(title="ReDay AI Memory Server (임시 로컬)")
def normalize_query(query: str) -> str:
    # 이름/단어 뒤 조사 제거
    query = re.sub(r'(이랑|랑|와|과|이와)\s*', ' ', query)

    # 공백 정리
    return ' '.join(query.split())

class RecordFragment(BaseModel):
    type: str                    # "TEXT", "PHOTO", "VOICE"
    content: str
    time: str                    # "2026-03-24T18:10:00"
    location: Optional[str] = None


class GenerateMemoryRequest(BaseModel):
    date: str                    # "2026-03-24"
    records: List[RecordFragment]
    photo_data: List[str] = []   # base64 인코딩된 사진 목록


class GenerateMemoryResponse(BaseModel):
    title: str
    summary: str
    tags: List[str]
    people: List[str] = []
    emotion: str = "😐 평범한"
    embedding: List[float] = []


#백엔드가 memoryID로 보내도 됨
class MemoryTextItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    memory_id: int = Field(alias="memoryId")
    text: str


class SearchSemanticRequest(BaseModel):
    query: str
    memories: List[MemoryTextItem]


class SearchSemanticResponse(BaseModel):
    ranked_ids: List[int]  # 유사도 높은 순


class MemorySummaryInput(BaseModel):
    date: str
    title: str
    summary: str
    tags: List[str] = []
    locations: List[str] = []
    people: List[str] = []


class GenerateInsightRequest(BaseModel):
    user_id: int
    year_month: str
    memories: List[MemorySummaryInput]


class GenerateInsightResponse(BaseModel):
    insight: str


class MemoryForComment(BaseModel):
    title: str
    summary: str
    tags: List[str] = []


class DailyCommentRequest(BaseModel):
    memories: List[MemoryForComment]


class DailyCommentResponse(BaseModel):
    comment: str


class ParseSearchRequest(BaseModel):
    query: str


class ParseSearchResponse(BaseModel):
    people: List[str] = []
    tags: List[str] = []
    locations: List[str] = []
    yearMonth: Optional[str] = None
    keywords: List[str] = []
    sentiment: Optional[str] = None


class LocationAnalysisMemoryInput(BaseModel):
    title: str
    summary: str
    locations: List[str] = []
    tags: List[str] = []


class LocationAnalysisRequest(BaseModel):
    memories: List[LocationAnalysisMemoryInput]


class PlaceStat(BaseModel):
    location: str
    count: int
    tags: List[str] = []


class LocationAnalysisResponse(BaseModel):
    top_places: List[str] = []
    place_stats: List[PlaceStat] = []
    analysis: str


@app.get("/health", summary="서버 상태 확인")
def health():
    return {"status": "ok", "mode": "claude-api"}


@app.post("/transcribe", summary="음성 STT 변환")
async def transcribe(file: UploadFile = File(...)):
    """음성 파일을 받아 Whisper로 텍스트 변환"""
    contents = await file.read()
    try:
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, contents, file.content_type),
            language="ko"
        )
        return {"text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {e}")


@app.post("/daily-comment", response_model=DailyCommentResponse, summary="오늘의 한마디 생성")
async def daily_comment(req: DailyCommentRequest):
    if not req.memories:
        raise HTTPException(status_code=400, detail="memories가 비어 있습니다")
    memories_dict = [m.model_dump() for m in req.memories]
    comment = generate_daily_comment(memories_dict)
    return DailyCommentResponse(comment=comment)


@app.post("/parse-search", response_model=ParseSearchResponse, summary="검색어 자연어 분석")
async def parse_search(req: ParseSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query가 비어 있습니다")
    result = parse_search_query(req.query)
    print(f"[parse-search] result: {result}")
    return ParseSearchResponse(**result)


def _compute_top_people(memories: list) -> list:
    counter = Counter()
    for m in memories:
        for person in m.get("people", []):
            if person:
                counter[person] += 1
    return [{"name": name, "count": count} for name, count in counter.most_common()]


def _compute_top_activities(memories: list) -> list:
    total = len(memories)
    if total == 0:
        return []
    counter = Counter()
    for m in memories:
        for tag in m.get("tags", []):
            if tag:
                counter[tag] += 1
    return [
        {"activityType": tag, "percentage": round(count / total * 100)}
        for tag, count in counter.most_common()
    ]


async def _save_insight_to_spring_boot(user_id: int, year: int, month: int, insight_text: str, top_people: list, top_activities: list):
    url = f"{SPRING_BOOT_BASE_URL}/api/analysis/monthly/insight"
    headers = {
        "X-Internal-Key": INTERNAL_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "userId": user_id,
        "year": year,
        "month": month,
        "insightText": insight_text,
        "topPeople": top_people,
        "topActivities": top_activities
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=body, headers=headers)
        if response.status_code != 204:
            raise RuntimeError(f"Spring Boot 저장 실패: {response.status_code} {response.text}")


@app.post("/generate-insight", response_model=GenerateInsightResponse, summary="월간 인사이트 생성")
async def generate_insight(req: GenerateInsightRequest):
    if not req.memories:
        raise HTTPException(status_code=400, detail="memories가 비어 있습니다")

    memories_dict = [m.model_dump() for m in req.memories]
    insight = generate_insight_with_ai(req.year_month, memories_dict)

    try:
        year, month = req.year_month.split("-")
        top_people = _compute_top_people(memories_dict)
        top_activities = _compute_top_activities(memories_dict)
        await _save_insight_to_spring_boot(
            user_id=req.user_id,
            year=int(year),
            month=int(month),
            insight_text=insight,
            top_people=top_people,
            top_activities=top_activities
        )
        print(f"[generate-insight] Spring Boot 저장 완료: {req.year_month}")
    except Exception as e:
        print(f"[generate-insight] Spring Boot 저장 실패 (인사이트는 반환): {e}")

    return GenerateInsightResponse(insight=insight)


@app.post("/generate-memory", response_model=GenerateMemoryResponse, summary="AI 기억 생성")
async def generate_memory(req: GenerateMemoryRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="records가 비어 있습니다")

    records_dict = [r.model_dump() for r in req.records]
    result = generate_memory_with_ai(records_dict, req.photo_data)

    embed_text = (
        f"{result['title']} "
        f"{result['summary']} "
        f"{' '.join(result['tags'])}"
    )
    try:
        embedding = generate_embedding(embed_text)
    except Exception:
        embedding = []

    return GenerateMemoryResponse(
        title=result["title"],
        summary=result["summary"],
        tags=result["tags"],
        people=result.get("people", []),
        emotion=result.get("emotion", "😐 평범한"),
        embedding=embedding
    )


#백엔드에서 memory_ids만 보내면 안됨. AI서버가 기억 내용 "직접" 봐야 키워드 가중치 줄 수 있음
@app.post("/search-semantic", response_model=SearchSemanticResponse, summary="의미 기반 기억 검색")
async def search_semantic(req: SearchSemanticRequest):
    """쿼리와 기억 텍스트들의 코사인 유사도로 순위 반환"""

    if not req.memories:
        return SearchSemanticResponse(ranked_ids=[])

    try:
        normalized_query = normalize_query(req.query)
        query_embedding = generate_embedding(normalized_query)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"쿼리 임베딩 생성 실패: {e}")

    def cosine_similarity(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    scored = []

    for item in req.memories:
        print(
            f"[memory] id={item.memory_id}, "
            f"text={item.text}"
        )

        if not item.text.strip():
            continue

        try:
            memory_embedding = generate_embedding(item.text)

            embedding_score = cosine_similarity(
                query_embedding,
                memory_embedding
            )

            bonus = keyword_bonus(normalized_query, item.text)
            final_score = embedding_score + bonus

            scored.append((item.memory_id, final_score))

            print(
                f"[semantic] id={item.memory_id}, "
                f"embedding={embedding_score:.4f}, "
                f"bonus={bonus:.2f}, "
                f"final={final_score:.4f}"
            )

        except Exception as e:
            print(
                f"[semantic] memory_id={item.memory_id} "
                f"embedding 실패: {e}"
            )

    scored.sort(key=lambda x: x[1], reverse=True)

    ranked_ids = [
        mid for mid, score in scored
        if score >= 0.35
    ]

    return SearchSemanticResponse(ranked_ids=ranked_ids)


@app.post("/analyze-locations", response_model=LocationAnalysisResponse, summary="장소 기반 기억 분석")
async def analyze_locations(req: LocationAnalysisRequest):
    if not req.memories:
        raise HTTPException(status_code=400, detail="memories가 비어 있습니다")

    memories_dict = [m.model_dump() for m in req.memories]
    result = analyze_location_patterns(memories_dict)

    return LocationAnalysisResponse(
        top_places=result.get("top_places", []),
        place_stats=result.get("place_stats", []),
        analysis=result.get("analysis", "")
    )


class AnalyzeTextRequest(BaseModel):
    memo_text: str = ""
    stt_text: str = ""


class AnalyzeTextResponse(BaseModel):
    summary: str
    main_topic: str
    emotion: str
    activity_hint: str


class ExtractKeywordsRequest(BaseModel):
    memo_text: str = ""
    stt_text: str = ""


class ExtractKeywordsResponse(BaseModel):
    keywords: List[str] = []


class ClassifyTagsRequest(BaseModel):
    text: str


class ClassifyTagsResponse(BaseModel):
    tags: List[str] = []


class ClassifyActivityRequest(BaseModel):
    text: str


class ClassifyActivityResponse(BaseModel):
    activity_type: str


class ExtractSearchKeywordsRequest(BaseModel):
    query: str


class ExtractSearchKeywordsResponse(BaseModel):
    keywords: List[str] = []


@app.post("/analyze-text", response_model=AnalyzeTextResponse, summary="기록 텍스트 분석")
async def analyze_text(req: AnalyzeTextRequest):
    if not req.memo_text.strip() and not req.stt_text.strip():
        raise HTTPException(status_code=400, detail="memo_text와 stt_text가 모두 비어 있습니다")

    result = analyze_record_text(req.memo_text, req.stt_text)

    return AnalyzeTextResponse(
        summary=result.get("summary", ""),
        main_topic=result.get("main_topic", ""),
        emotion=result.get("emotion", "중립"),
        activity_hint=result.get("activity_hint", "")
    )


@app.post("/extract-keywords", response_model=ExtractKeywordsResponse, summary="기록 기반 키워드 추출")
async def extract_keywords(req: ExtractKeywordsRequest):
    if not req.memo_text.strip() and not req.stt_text.strip():
        raise HTTPException(status_code=400, detail="memo_text와 stt_text가 모두 비어 있습니다")

    result = extract_record_keywords(req.memo_text, req.stt_text)

    return ExtractKeywordsResponse(
        keywords=result.get("keywords", [])
    )


@app.post("/classify-tags", response_model=ClassifyTagsResponse, summary="태그 기반 기억 분류")
async def classify_tags(req: ClassifyTagsRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어 있습니다")

    result = classify_memory_tags(req.text)

    return ClassifyTagsResponse(
        tags=result.get("tags", [])
    )


@app.post("/classify-activity", response_model=ClassifyActivityResponse, summary="활동 유형 분류")
async def classify_activity(req: ClassifyActivityRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어 있습니다")

    result = classify_activity_type(req.text)

    return ClassifyActivityResponse(
        activity_type=result.get("activity_type", "")
    )


@app.post("/extract-search-keywords", response_model=ExtractSearchKeywordsResponse, summary="검색어 기반 키워드 추출")
async def extract_search_keywords_api(req: ExtractSearchKeywordsRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query가 비어 있습니다")

    result = extract_search_keywords(req.query)

    return ExtractSearchKeywordsResponse(
        keywords=result.get("keywords", [])
    )
