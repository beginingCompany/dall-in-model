"""
Test script to verify the personality analyzer API endpoint
"""
import requests
import json

def test_analyze_personality_endpoint():
    """Test the /analyze-personality endpoint with different inputs."""
    base_url = "http://localhost:8000"
    
    print("=" * 80)
    print("TESTING API ENDPOINT")
    print("=" * 80)
    
    print("\nTest 1: Basic input (should trigger clarification questions)")
    payload = {
        "id": 123,
        "user_input": "I am a software developer",
        "languages": ["en"]
    }
    
    try:
        response = requests.post(f"{base_url}/analyze-personality", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status')}")
            print(f"Description: {result.get('description_english')}")
            print(f"Questions: {result.get('clarification_questions')}")
            print("-" * 50)
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    print("\nTest 2: More detailed input")
    payload = {
        "id": 456,
        "user_input": """I'm a software developer with 5 years of experience. I enjoy solving complex problems
        and working with my team to create innovative solutions. I'm usually calm under pressure
        but can get excited when discovering a new solution. I prefer to plan my work carefully
        and stick to schedules.""",
        "languages": ["en"]
    }
    
    try:
        response = requests.post(f"{base_url}/analyze-personality", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status')}")
            print(f"Description: {result.get('description_english')[:100]}..." if result.get('description_english') else "No description")
            print(f"Questions: {result.get('clarification_questions')}")
            print("-" * 50)
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
        
    print("\nTest 3: Testing response with a question in description")
    payload = {
        "id": 789,
        "user_input": "I write code",
        "languages": ["en"]
    }
    
    try:
        response = requests.post(f"{base_url}/analyze-personality", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status')}")
            print(f"Description: {result.get('description_english')}")
            print(f"Questions: {result.get('clarification_questions')}")
            print("-" * 50)
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_analyze_personality_endpoint()
