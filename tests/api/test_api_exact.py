#!/usr/bin/env python3
"""
Test the exact API request scenario
"""

import requests
import json

def test_api_exact_scenario():
    """Test the exact scenario through API endpoint"""
    
    print("🧪 TESTING EXACT SCENARIO VIA API")
    print("=" * 40)
    
    # Exact data from user's request
    test_data = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions. "
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "انا المهندس احمد"
            }
        ],
        "languages": "ar"
    }
    
    # API endpoint
    url = "http://127.0.0.1:8000/analyze-personality"
    
    print("Test data:")
    print(f"  Last answer: '{test_data['new_input'][-1]['answer']}'")
    print("Calling API...")
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Status: {result.get('status')}")
            print(f"Personal Greeting: '{result.get('personal_greeting', '')}'")
            
            greeting = result.get('personal_greeting', '')
            if greeting and len(greeting.strip()) > 0:
                print("\n✅ SUCCESS: API returns personal greeting!")
                print(f"   Greeting: {greeting}")
            else:
                print("\n❌ FAILED: API personal greeting is empty")
                
        else:
            print(f"❌ API Error: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure the server is running: uvicorn app.api:app --reload")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    test_api_exact_scenario()
