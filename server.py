"""
[임시 로컬 서버] FastAPI + Claude API
나중에 실서버로 옮길 때 BASE_URL, 인증, DB 연결만 교체하면 됨
"""

from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from memory_generator import generate_memory_with_ai
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI(title="ReDay AI Memory Server (임시 로컬)")


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


@app.get("/health")
def health():
    return {"status": "ok", "mode": "claude-api"}


@app.post("/transcribe")
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


@app.post("/generate-memory", response_model=GenerateMemoryResponse)
async def generate_memory(req: GenerateMemoryRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="records가 비어 있습니다")

    records_dict = [r.model_dump() for r in req.records]
    result = generate_memory_with_ai(records_dict, req.photo_data)

    return GenerateMemoryResponse(
        title=result["title"],
        summary=result["summary"],
        tags=result["tags"],
        people=result.get("people", [])
    )
