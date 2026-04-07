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
        display_hour = hour if 0 < hour <= 12 else (12 if hour == 0 else hour - 12)
        return f"{period} {display_hour}시 {minute:02d}분" if minute else f"{period} {display_hour}시"
    except Exception:
        return time_str


ALLOWED_TAGS = ["여행", "카페", "산책", "공부", "운동", "유흥", "자연", "문화생활", "쇼핑", "식사", "휴식"]


def build_daily_comment_prompt(memories: list) -> str:
    memory_lines = []
    for m in memories:
        tags_str = ", ".join(m.get("tags", [])) or "없음"
        memory_lines.append(f"- {m.get('title', '')} ({tags_str}): {m.get('summary', '')}")

    joined = "\n".join(memory_lines)

    prompt = f"""너는 사용자의 최근 기억을 보고 따뜻한 한마디를 건네는 AI야.

최근 기억:
{joined}

규칙:
1. 최근 기억의 패턴이나 분위기를 자연스럽게 반영해서 한마디를 건네줘.
2. 1~2문장으로 짧고 따뜻하게. 친구처럼 편하게.
3. 오늘 하루를 응원하거나 최근 기억을 살짝 언급해도 좋아.
4. 없는 내용은 만들지 마.
5. 반드시 JSON 형식으로만 응답.

좋은 예시:
- "요즘 카페를 자주 가시네요. 오늘도 여유로운 하루 보내세요 ☕"
- "지수님과 함께한 날들이 많았네요. 소중한 인연을 잘 챙기고 계신 것 같아요 😊"

출력 형식:
{{"comment": "한마디 텍스트"}}"""

    return prompt


import datetime as _dt


def build_search_prompt(query: str) -> str:
    allowed_tags = ["여행", "카페", "산책", "공부", "운동", "유흥", "자연", "문화생활", "쇼핑", "식사", "휴식"]
    allowed_tags_str = ", ".join(allowed_tags)
    current_year = _dt.date.today().year

    prompt = f"""너는 사용자의 기억 검색 의도를 분석하는 AI야.

사용자 입력: "{query}"

아래 규칙에 따라 검색 의도를 JSON으로 파싱해줘.

규칙:
1. people: 함께한 사람 이름 목록. "이랑", "와", "랑", "이와" 등 조사 제거 후 이름만. 없으면 []
2. tags: 다음 목록에서만 선택. 없으면 []
   - 여행: 여행/나들이
   - 카페: 카페/커피
   - 산책: 걷기/산책
   - 공부: 학습/독서/업무
   - 운동: 헬스/스포츠
   - 유흥: 술/맥주/소주/피맥/치맥/클럽/노래방/게임/리그오브레전드/롤/배그 등 오락
   - 자연: 공원/숲/바다/산
   - 문화생활: 영화/공연/전시
   - 쇼핑: 쇼핑/마트
   - 식사: 밥/음식점/식사 (술은 유흥)
   - 휴식: 집/쉬기/낮잠
3. locations: 장소 키워드 목록. 없으면 []
4. yearMonth: "yyyy-MM" 형식. 특정 달이 언급되면 현재 연도({current_year}) 기준. 없으면 null
5. keywords: 위 항목에 해당하지 않는 나머지 검색 키워드. 없으면 []
6. sentiment: 감정 방향. 아래 중 하나. 명확한 감정 표현이 없으면 null
   - "긍정": 좋은/행복한/즐거운/설레는/신나는/재밌는/좋았던/놀았던/같이한/함께한/즐겼던/웃었던 등
   - "부정": 힘든/슬픈/속상한/안좋은/지친/다퉜던/싸웠던/힘들었던/울었던/화났던 등
7. 반드시 JSON 형식으로만 응답. 다른 말 하지 마.

예시:
- "지수랑 카페 갔던 기억" → {{"people": ["지수"], "tags": ["카페"], "locations": [], "yearMonth": null, "keywords": [], "sentiment": null}}
- "3월에 한강 산책" → {{"people": [], "tags": ["산책"], "locations": ["한강"], "yearMonth": "2026-03", "keywords": [], "sentiment": null}}
- "진성이와 안좋았던 기억" → {{"people": ["진성"], "tags": [], "locations": [], "yearMonth": null, "keywords": [], "sentiment": "부정"}}
- "카리나랑 행복했던 날" → {{"people": ["카리나"], "tags": [], "locations": [], "yearMonth": null, "keywords": [], "sentiment": "긍정"}}

출력 형식:
{{"people": [], "tags": [], "locations": [], "yearMonth": null, "keywords": [], "sentiment": null}}"""

    return prompt


def build_insight_prompt(year_month: str, memories: list) -> str:
    try:
        year, month = year_month.split("-")
        month_label = f"{int(month)}월"
    except Exception:
        month_label = year_month

    memory_lines = []
    for m in memories:
        tags_str = ", ".join(m.get("tags", [])) or "없음"
        locations_str = ", ".join(m.get("locations", [])) or "없음"
        people_str = ", ".join(m.get("people", [])) or "없음"
        memory_lines.append(
            f"- [{m.get('date', '')}] {m.get('title', '')}\n"
            f"  요약: {m.get('summary', '')}\n"
            f"  태그: {tags_str} | 장소: {locations_str} | 함께한 사람: {people_str}"
        )

    joined_memories = "\n".join(memory_lines)

    prompt = f"""너는 사용자의 한 달 기억들을 따뜻하게 돌아봐주는 AI야.

아래는 {month_label}에 기록된 기억 목록이야:

{joined_memories}

규칙:
1. 이 기억들을 바탕으로 {month_label}을 돌아보는 인사이트 텍스트를 작성해.
2. 이 달에 자주 등장한 장소, 사람, 활동 패턴을 자연스럽게 녹여서 써줘.
3. 따뜻하고 감성적인 톤으로, 마치 친한 친구가 한 달을 정리해주는 느낌으로.
4. 3~4문장 이내로 간결하게.
5. 없는 내용은 절대 만들지 마. 기억에 있는 내용만 사용해.
6. 반드시 JSON 형식으로만 응답. 다른 말 하지 마.

좋은 예시:
- "이번 달에는 산책과 힐링에 집중한 한 달이었네요. 특히 한강공원을 자주 방문하셨고, 친구 지수님과 많은 시간을 보냈어요."

출력 형식:
{{"insight": "인사이트 텍스트"}}"""

    return prompt


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
1. 입력된 기록 텍스트와 함께 제공된 사진(있는 경우)을 모두 활용해. 사진에서 보이는 구체적인 내용(음식 이름, 장소 특징, 분위기 등)을 summary에 자연스럽게 녹여줘. 예: 순대국밥 사진이 있으면 "순대국밥 한 그릇"처럼 구체적으로. 단, 사진에서 명확히 보이는 것만 언급하고 불확실한 내용은 만들지 마.
2. 기록과 사진에 없는 내용은 절대 창작하지 마.
3. title은 하루 전체의 분위기나 핵심을 담은 짧고 감성적인 한국어 제목 1개 (10자 이내). 단순 사실 나열 금지.
4. summary는 단순히 기록을 나열하지 말고, 그날의 감정과 분위기가 느껴지도록 2~3문장으로 써줘. 마치 나중에 다시 읽고 싶은 일기처럼. 문장 끝은 자연스럽게 통일해.
5. tags는 다음 목록에서만 선택해 (기록 내용에 맞는 것만, 1~5개): {allowed_tags_str}
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
6. emotion은 하루 전체 흐름을 고려해서 가장 기억에 남을 감정 1개를 다음 목록에서 정확히 선택:
   "😊 즐거운" / "🥰 설레는" / "😌 평온한" / "😤 지친" / "😢 힘든" / "🤩 신나는" / "😐 평범한"
   판단 기준:
   - 감정적 피크(가장 강렬한 감정)를 우선으로 선택
   - 아침 피곤 + 저녁 신남 → "🤩 신나는" (끝이 좋으면 전체가 좋은 법)
   - 감정 단서가 아예 없으면 → "😐 평범한"
   - 반드시 위 목록 중 하나를 이모지+공백+한글 형식 그대로 출력 (예: "😊 즐거운")
7. people은 사용자가 실제로 함께 있었거나 같이 행동한 사람 이름만 포함. 대화 주제로만 언급된 사람(뒷담, 이야기 속 인물 등)은 절대 포함하지 마. '그룹명', '이름', '누군가', '친구' 같은 플레이스홀더나 모호한 단어 절대 사용 금지. 함께한 사람이 없으면 반드시 빈 배열 []
   판단 기준:
   - 포함 O: "진성이랑 카페 갔다", "민준이와 밥 먹었다" → ["진성", "민준"]
   - 포함 X: "주희 뒷담했다", "철수 얘기했다" → 주희, 철수는 제외
8. [시간 흐름 연속성 - summary 작성 시] 앞선 기록에서 함께한 사람이 이후 기록에서 명시되지 않더라도, 시간상 자연스럽게 이어지는 흐름이면 같이 있었을 가능성을 summary에 자연스럽게 반영해도 좋아. 단, people 배열 값은 이 규칙의 영향을 받지 않으며 7번 규칙만 따른다.
9. 반드시 JSON 형식으로만 응답. 다른 말 하지 마.

좋은 summary 예시:
- "주희와 함께한 치맥 한 잔, 로또와의 한강 산책까지. 사람과 동물로 가득했던 하루였어요."
- "조용히 혼자 보낸 하루였지만, 커피 한 잔과 함께한 공부가 꽤 뿌듯했어요."

나쁜 summary 예시 (하지 말 것):
- "오후에는 치맥을 하였고, 저녁에는 산책을 했다." (단순 나열)

출력 형식:
{{"title": "제목", "summary": "요약", "tags": ["태그1", "태그2"], "people": ["이름1", "이름2"], "emotion": "😊 즐거운"}}"""

    return prompt
