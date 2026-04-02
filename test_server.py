import requests

res = requests.post("http://localhost:8000/generate-memory", json={
    "date": "2026-03-24",
    "records": [
        {"type": "PHOTO", "content": "한강에서 찍은 노을 사진", "time": "2026-03-24T18:10:00", "location": "서울 한강공원"},
        {"type": "TEXT",  "content": "오랜만에 친구랑 산책해서 기분이 좋았다.", "time": "2026-03-24T18:30:00", "location": "서울 한강공원"},
        {"type": "VOICE", "content": "오늘은 바람이 좀 불었지만 날씨가 좋아서 걷기 좋았다.", "time": "2026-03-24T18:45:00", "location": "서울 한강공원"},
    ]
})

print(res.status_code)
print(res.json())
