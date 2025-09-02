#!/usr/bin/env python3
"""
Test the identity response system with the actual API endpoint
"""

import requests
import json
import time

def test_api_identity_system():
    """Test the identity system through the actual API"""
    
    API_BASE_URL = "http://localhost:8000"  # Adjust if your API runs on a different port
    
    print("Testing Identity System with Live API")
    print("=" * 50)
    
    # Test case 1: Identity question in English
    test_case_1 = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles."
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 1: Identity Question in English")
    print(f"Sending request to {API_BASE_URL}/analyze-personality")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-personality",
            json=test_case_1,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status Code: {response.status_code}")
            print(f"Response Status: {data.get('status')}")
            print(f"ID: {data.get('id')}")
            print(f"Identity Response: {data.get('description_identity', 'N/A')[:150]}...")
            print(f"English Description: {data.get('description_english', 'EMPTY')}")
            print(f"Arabic Description: {data.get('description_arabic', 'EMPTY')}")
            print(f"Missing Traits: {data.get('missing_traits', [])}")
            print(f"Clarification Questions: {data.get('clarification_questions', [])}")
            
            # Verify expected behavior
            if data.get('status') == 'identity':
                print("✅ Test Case 1: PASSED - Identity status detected")
            else:
                print(f"❌ Test Case 1: FAILED - Expected 'identity', got '{data.get('status')}'")
        else:
            print(f"❌ Test Case 1: FAILED - Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Case 1: FAILED - Request error: {str(e)}")
        print("Make sure the API server is running on localhost:8000")
        return
    
    print("\n" + "-" * 40 + "\n")
    
    # Test case 2: Non-identity question (should proceed normally)
    test_case_2 = {
        "id": 225985882207,
        "user_input": "Hello! I'm someone who really enjoys working with data.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and mentoring colleagues."
            },
            {
                "question": "How do you handle emotions?",
                "answer": "I try to stay calm and think through problems logically."
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 2: Non-Identity Question (Normal Processing)")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-personality",
            json=test_case_2,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status Code: {response.status_code}")
            print(f"Response Status: {data.get('status')}")
            print(f"ID: {data.get('id')}")
            
            # Should NOT be identity status
            if data.get('status') != 'identity':
                print("✅ Test Case 2: PASSED - Normal processing (not identity)")
                print(f"Status: {data.get('status')}")
                if data.get('clarification_questions'):
                    print(f"Clarification Questions: {len(data.get('clarification_questions', []))} questions")
            else:
                print(f"❌ Test Case 2: FAILED - Got 'identity' status when not expected")
        else:
            print(f"❌ Test Case 2: FAILED - Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Case 2: FAILED - Request error: {str(e)}")
        return
    
    print("\n" + "-" * 40 + "\n")
    
    # Test case 3: Arabic identity question
    test_case_3 = {
        "id": 225985882208,
        "user_input": "أنا شخص يحب العمل مع البيانات",
        "new_input": [
            {
                "question": "كيف تتفاعل مع الآخرين؟",
                "answer": "من أنت"
            }
        ],
        "languages": "ar"
    }
    
    print("Test Case 3: Arabic Identity Question")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-personality",
            json=test_case_3,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status Code: {response.status_code}")
            print(f"Response Status: {data.get('status')}")
            print(f"ID: {data.get('id')}")
            
            if data.get('status') == 'identity':
                identity_response = data.get('description_identity', '')
                print(f"Identity Response: {identity_response[:150]}...")
                
                # Check if it contains Arabic text
                has_arabic = any('\u0600' <= char <= '\u06FF' for char in identity_response)
                if has_arabic:
                    print("✅ Test Case 3: PASSED - Arabic identity response detected")
                else:
                    print("⚠️ Test Case 3: WARNING - No Arabic text in identity response")
            else:
                print(f"❌ Test Case 3: FAILED - Expected 'identity', got '{data.get('status')}'")
        else:
            print(f"❌ Test Case 3: FAILED - Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Case 3: FAILED - Request error: {str(e)}")
        return
    
    print("\n🎉 API Identity System Testing Complete!")

def test_multiple_identity_triggers():
    """Test various identity trigger phrases"""
    
    API_BASE_URL = "http://localhost:8000"
    
    print("\n\nTesting Multiple Identity Triggers")
    print("=" * 50)
    
    identity_triggers = [
        "who are you",
        "tell me about you",
        "introduce yourself", 
        "what is begining",
        "explain begining",
        "who is your developer",
        "who made you",
        "what is your purpose",
        "how do you work"
    ]
    
    for i, trigger in enumerate(identity_triggers, 1):
        print(f"\nTrigger {i}: '{trigger}'")
        
        test_case = {
            "id": 999900 + i,
            "user_input": "I'm a test user",
            "new_input": [
                {
                    "question": "Test question",
                    "answer": trigger
                }
            ],
            "languages": "en"
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze-personality",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'identity':
                    print(f"✅ Identity detected: {data.get('description_identity', '')[:80]}...")
                else:
                    print(f"❌ Identity NOT detected - Status: {data.get('status')}")
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {str(e)}")
            break
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)

if __name__ == "__main__":
    print("Make sure the API server is running with: uvicorn app.api:app --reload")
    print("Press Enter to continue with the tests...")
    input()
    
    test_api_identity_system()
    test_multiple_identity_triggers()
