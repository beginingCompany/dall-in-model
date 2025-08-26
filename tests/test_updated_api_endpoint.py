"""
Test the actual API endpoint with the user's exact scenario - Updated version
"""

import requests
import json

def test_api_endpoint_updated():
    """Test the actual API endpoint with the problematic scenario."""
    
    print("🌐 TESTING ACTUAL API ENDPOINT (UPDATED)")
    print("=" * 50)
    
    # API endpoint
    url = "http://127.0.0.1:8000/analyze-personality"
    
    # Exact request data from user that was showing identity response
    request_data = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        "new_input": [
            {
                "question": "How do you usually interact with others in group settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "who are you"
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "i analytical can solving the problems by analyze them "
            }
        ],
        "languages": "en"
    }
    
    print("REQUEST:")
    print(json.dumps(request_data, indent=2))
    print()
    
    try:
        # Make API call
        response = requests.post(url, json=request_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API RESPONSE:")
            print(json.dumps(result, indent=2))
            print()
            
            # Analysis
            has_identity = bool(result.get('description_identity'))
            print("📊 ANALYSIS:")
            print(f"Status Code: {response.status_code}")
            print(f"Has identity response: {has_identity}")
            
            if has_identity:
                print("❌ PROBLEM: Identity response still triggered in API!")
                print(f"   Response: {result['description_identity']}")
                print("   👉 This means the API is still using old code or there's a caching issue")
            else:
                print("✅ SUCCESS: No identity response (correctly filtered)")
                print("   👉 API is now using the updated context-aware code")
            
            print(f"Status: {result.get('status')}")
            print(f"Missing traits: {result.get('missing_traits', [])}")
            print(f"Clarification questions: {len(result.get('clarification_questions', []))} questions")
            
        else:
            print(f"❌ API Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_api_endpoint_updated()
