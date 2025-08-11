import requests
import json

url = "http://localhost:8000/analyze-personality"

payload = {
    "id": 12345,
    "user_input": "I'm a software developer who enjoys working on AI projects. I prefer to work alone but can collaborate with others when needed.",
        "new_input": [
        {
            "question": "Could you tell me more about how you interact with others in your professional environment?",
            "answer": "I interact with others in my professional environment mainly through team meetings and code reviews"
        }
    ],

    "languages": ["en"]
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print("Error:", e)
