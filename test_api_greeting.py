#!/usr/bin/env python3
"""
Test the API endpoint to ensure greeting functionality works via HTTP calls.
"""

import requests
import json
import time

def test_api_greeting():
    """Test the API endpoint with Arabic introduction"""
    
    # Start the API server first (if not already running)
    base_url = "http://localhost:8000"
    
    print("Testing API Greeting Functionality")
    print("=" * 50)
    
    # Test case 1: Arabic introduction
    print("\n1. Testing Arabic Introduction via API")
    
    payload1 = {
        "id": 1,
        "user_input": "مرحبا انا وليد مهندس بيوميجات",
        "new_input": [],
        "languages": "ar"
    }
    
    try:
        response1 = requests.post(
            f"{base_url}/analyze-personality",
            json=payload1,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"Status: {result1.get('status', 'N/A')}")
            print(f"Personal Greeting: '{result1.get('personal_greeting', '')}'")
            print(f"Missing Traits: {result1.get('missing_traits', [])}")
            
            if result1.get('personal_greeting'):
                print("✅ API returned personal greeting!")
            else:
                print("❌ API did not return personal greeting")
                
        else:
            print(f"❌ API request failed with status: {response1.status_code}")
            print(f"Response: {response1.text}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running. Start the server with:")
        print("uvicorn app.api:app --reload --host 0.0.0.0 --port 8000")
        return
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        return
    
    # Test case 2: Follow-up identity question
    print("\n2. Testing Follow-up Identity Question")
    
    first_question = result1.get('clarification_questions', [''])[0] if 'result1' in locals() else "كيف تشعر في المواقف الصعبة؟"
    
    payload2 = {
        "id": 1,
        "user_input": "مرحبا انا وليد مهندس بيوميجات",
        "new_input": [
            {"question": first_question, "answer": "من أنت"}
        ],
        "languages": "ar"
    }
    
    try:
        response2 = requests.post(
            f"{base_url}/analyze-personality",
            json=payload2,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"Status: {result2.get('status', 'N/A')}")
            print(f"Personal Greeting: '{result2.get('personal_greeting', '')}'")
            
            if result2.get('status') == 'identity':
                print(f"Identity Response: '{result2.get('description_identity', '')[:100]}...'")
                print("✅ Identity question handled properly via API")
            else:
                print("❌ Identity question not detected via API")
                
        else:
            print(f"❌ API request failed with status: {response2.status_code}")
            
    except Exception as e:
        print(f"❌ Error calling API: {e}")

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running and healthy")
            return True
        else:
            print(f"⚠️  API server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running")
        return False
    except Exception as e:
        print(f"❌ Error checking API health: {e}")
        return False

if __name__ == "__main__":
    print("Checking API Server Status...")
    if test_api_health():
        test_api_greeting()
    else:
        print("\nTo start the API server, run:")
        print("uvicorn app.api:app --reload --host 0.0.0.0 --port 8000")
