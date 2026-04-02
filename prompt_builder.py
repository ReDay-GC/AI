def _format_time(time_str: str) -> str:
    try:
        hour = int(time_str[11:13])
        minute = int(time_str[14:16])
        if 5 <= hour < 12:
            period = "오전"
        elif 12 <= hour < 17:
            period = "오후"
        elif 17 <= hour < 21:
            period = "저녁"
        else:
            period = "밤"
        display_hour = hour if hour <= 12 else hour - 12
        return f"{period} {display_hour}시 {minute:02d}분" if minute else f"{period} {display_hour}시"
    except Exception:
        return time_str


ALLOWED_TAGS = ["여행", "카페", "산책", "공부", "운동", "유흥", "자연", "문화생활", "쇼핑", "식사", "휴식"]


def build_memory_prompt(records):
    record_lines = []

    for idx, record in enumerate(records, start=1):
        record_lines.append(
            f"{idx}. "
            f"type={record['type']}, "
            f"time={_format_time(record['time'])}, "
            f"location={record['location']}, "
            f"content={record['content']}"
        )

    joined_records = "\n".join(record_lines)
    allowed_tags_str = ", ".join(ALLOWED_TAGS)

    prompt = f"""너는 사용자의 하루 기록 조각들을 분석해서 하나의 따뜻한 '기억 카드'를 만드는 AI다.

입력 기록:
{joined_records}

규칙:
1. 입력된 기록에 있는 내용만 사용해. 없는 내용은 절대 만들지 마.
2. title은 하루 전체의 분위기나 핵심을 담은 짧고 감성적인 한국어 제목 1개 (10자 이내). 단순 사실 나열 금지.
3. summary는 단순히 기록을 나열하지 말고, 그날의 감정과 분위기가 느껴지도록 2~3문장으로 써줘. 마치 나중에 다시 읽고 싶은 일기처럼. 문장 끝은 자연스럽게 통일해.
4. tags는 다음 목록에서만 선택해 (기록 내용에 맞는 것만, 1~5개): {allowed_tags_str}
   각 태그 기준:
   - 여행: 일상 반경을 벗어난 여행/나들이
   - 카페: 카페/커피숍 방문 (술집·PC방·식당은 카페 아님)
   - 산책: 걷기/산책/자연 속 이동
   - 공부: 학습/독서/업무
   - 운동: 헬스/스포츠/체육 활동
   - 유흥: 술집/클럽/PC방/노래방/게임 등 오락 활동
   - 자연: 공원/숲/바다/산 등 자연 공간
   - 문화생활: 영화/공연/전시/뮤지컬
   - 쇼핑: 쇼핑/마트/시장
   - 식사: 음식점/식사 (카페는 식사 아님)
   - 휴식: 집에서 쉬기/낮잠/휴가
5. people은 사용자가 실제로 함께 있었거나 같이 행동한 사람 이름만 포함. 대화 주제로만 언급된 사람(뒷담, 이야기 속 인물 등)은 절대 포함하지 마. '그룹명', '이름', '누군가', '친구' 같은 플레이스홀더나 모호한 단어 절대 사용 금지. 함께한 사람이 없으면 반드시 빈 배열 []
   판단 기준:
   - 포함 O: "진성이랑 카페 갔다", "민준이와 밥 먹었다" → 진성, 민준
   - 포함 X: "주희 뒷담했다", "철수 얘기했다" → 주희, 철수는 제외
6. 반드시 JSON 형식으로만 응답. 다른 말 하지 마.

좋은 summary 예시:
- "주희와 함께한 치맥 한 잔, 로또와의 한강 산책까지. 사람과 동물로 가득했던 하루였어요."
- "조용히 혼자 보낸 하루였지만, 커피 한 잔과 함께한 공부가 꽤 뿌듯했어요."

나쁜 summary 예시 (하지 말 것):
- "오후에는 치맥을 하였고, 저녁에는 산책을 했다." (단순 나열)

출력 형식:
{{"title": "제목", "summary": "요약", "tags": ["태그1", "태그2"], "people": ["이름1", "이름2"]}}"""

    return prompt
