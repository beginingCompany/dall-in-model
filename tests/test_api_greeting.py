#!/usr/bin/env python3
"""
Test personal greeting via API
"""

import requests
import json

def test_api_greeting():
    """Test personal greeting through API endpoint"""
    
    print("🧪 TESTING PERSONAL GREETING VIA API")
    print("=" * 40)
    
    # Test data
    test_data = {
        "id": 225985882206,
        "user_input": "انا المهندس احمد",
        "new_input": [],
        "languages": "ar"
    }
    
    # API endpoint
    url = "http://127.0.0.1:8000/analyze-personality"
    
    print(f"Testing input: '{test_data['user_input']}'")
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
    test_api_greeting()
