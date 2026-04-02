from memory_generator import generate_memory

records = [
    {
        "type": "photo",
        "content": "한강에서 찍은 노을 사진",
        "time": "2026-03-24 18:10",
        "location": "서울 한강공원"
    },
    {
        "type": "text",
        "content": "오랜만에 친구랑 산책해서 기분이 좋았다.",
        "time": "2026-03-24 18:30",
        "location": "서울 한강공원"
    },
    {
        "type": "voice",
        "content": "오늘은 바람이 좀 불었지만 날씨가 좋아서 걷기 좋았다.",
        "time": "2026-03-24 18:45",
        "location": "서울 한강공원"
    }
]

result = generate_memory(records)

print("제목:", result["title"])
print("요약:", result["summary"])
print("태그:", result["tags"])