#!/usr/bin/env python3
"""
Test the API to confirm model-based identity detection is working correctly
"""

import requests
import json

def test_api_identity_detection():
    """Test API with various identity questions"""
    
    api_url = "http://127.0.0.1:8000/analyze-personality"
    
    test_cases = [
        {"input": "who are you", "language": "en"},
        {"input": "مين مطورك", "language": "ar"},
        {"input": "what is your purpose", "language": "en"},
        {"input": "I am a creative person", "language": "en"},
    ]
    
    print("Testing API with Model-based Identity Detection")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            payload = {
                "id": i,
                "user_input": test_case["input"],
                "new_input": [],
                "languages": test_case["language"]
            }
            
            response = requests.post(api_url, json=payload)
            result = response.json()
            
            identity_response = result.get("description_identity", "None")
            has_identity = "Yes" if identity_response != "None" and identity_response else "No"
            
            print(f"{i}. Input: '{test_case['input']}'")
            print(f"   Language: {test_case['language']}")
            print(f"   Has Identity Response: {has_identity}")
            if has_identity == "Yes":
                print(f"   Identity: {identity_response[:100]}...")
            print()
            
        except Exception as e:
            print(f"{i}. ERROR with '{test_case['input']}': {str(e)}")
    
    print("API test completed!")

if __name__ == "__main__":
    test_api_identity_detection()
