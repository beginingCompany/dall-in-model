#!/usr/bin/env python3

import requests
import json

def test_api_identity_questions():
    """Test the API with various identity questions in English and Arabic"""
    
    url = "http://127.0.0.1:8000/analyze-personality"
    
    test_cases = [
        # English questions
        {
            "id": 12345,
            "user_input": "Who are you?",
            "languages": "en",
            "expected_lang": "english"
        },
        {
            "id": 12346,
            "user_input": "Who made you?",
            "languages": "en", 
            "expected_lang": "english"
        },
        {
            "id": 12347,
            "user_input": "What's your purpose?",
            "languages": "en",
            "expected_lang": "english"
        },
        
        # Arabic questions
        {
            "id": 12348,
            "user_input": "من أنت؟",
            "languages": "ar",
            "expected_lang": "arabic"
        },
        {
            "id": 12349,
            "user_input": "مين عملك؟",
            "languages": "ar",
            "expected_lang": "arabic"
        },
        {
            "id": 12350,
            "user_input": "ايش هدفك؟",
            "languages": "ar",
            "expected_lang": "arabic"
        },
        
        # Variations and different phrasings
        {
            "id": 12351,
            "user_input": "Tell me about yourself",
            "languages": "en",
            "expected_lang": "english"
        },
        {
            "id": 12352,
            "user_input": "عرفني على نفسك",
            "languages": "ar",
            "expected_lang": "arabic"
        },
        {
            "id": 12353,
            "user_input": "How do you analyze personality?",
            "languages": "en",
            "expected_lang": "english"
        },
        {
            "id": 12354,
            "user_input": "كيف تحلل الشخصية؟",
            "languages": "ar",
            "expected_lang": "arabic"
        },
    ]
    
    print("=== Testing API Identity Detection ===\n")
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{total_count}: {test_case['user_input']}")
        
        try:
            response = requests.post(url, json=test_case)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if description_identity is present and not null
                if data.get("description_identity"):
                    print(f"✅ Identity response found: {data['description_identity'][:80]}...")
                    success_count += 1
                else:
                    print(f"❌ No identity response found. Response: {data}")
                    
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            
        print("-" * 70)
    
    print(f"\n=== Results ===")
    print(f"Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # Test language auto-detection
    print(f"\n=== Testing Language Auto-Detection ===")
    
    auto_tests = [
        {
            "id": 99991,
            "user_input": "Who are you?",  # English question
            "languages": "ar",  # Request Arabic but input is English
        },
        {
            "id": 99992,
            "user_input": "من أنت؟",  # Arabic question  
            "languages": "en",  # Request English but input is Arabic
        }
    ]
    
    for test in auto_tests:
        print(f"Testing: '{test['user_input']}' (Requested: {test['languages']})")
        
        try:
            response = requests.post(url, json=test)
            if response.status_code == 200:
                data = response.json()
                identity = data.get("description_identity", "")
                if identity:
                    lang_detected = "Arabic" if any('\u0600' <= c <= '\u06FF' for c in identity) else "English"
                    print(f"✅ Response language: {lang_detected}")
                    print(f"Response: {identity[:80]}...")
                else:
                    print(f"❌ No identity response")
            else:
                print(f"❌ API Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_api_identity_questions()
