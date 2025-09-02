#!/usr/bin/env python3
"""
Test script to verify the off-topic detection fix for the personality analyzer API.
"""

import requests
import json

def test_off_topic_scenario():
    """Test the exact scenario described in the user request"""
    
    # The request from the user
    test_request = {
        "id": 102,
        "user_input": "مرحبا! انا احمد كيف حالك؟",
        "new_input": [
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "من انت وما هدفك؟"
            },
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "ما لون السماء"
            }
        ],
        "languages": "ar"
    }
    
    print("Testing off-topic detection fix...")
    print("Request:")
    print(json.dumps(test_request, ensure_ascii=False, indent=2))
    print("\n" + "="*50 + "\n")
    
    try:
        # Make the API request
        response = requests.post(
            "http://localhost:8000/analyze",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # Check if the response correctly identifies as off-topic
            if result.get("personal_greeting_and_off_topic"):
                print("\n✅ SUCCESS: API correctly detected off-topic content!")
                print(f"Off-topic response: {result.get('personal_greeting_and_off_topic')}")
                
                # Verify that description_identity is None or empty
                if not result.get("description_identity"):
                    print("✅ SUCCESS: description_identity is correctly None")
                else:
                    print("❌ ISSUE: description_identity should be None for off-topic")
                
            else:
                print("\n❌ ISSUE: API did not properly detect off-topic content")
                print("Expected: personal_greeting_and_off_topic should be populated")
                print(f"Actual: personal_greeting_and_off_topic = '{result.get('personal_greeting_and_off_topic')}'")
                
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not connect to API: {e}")
        print("Make sure the API server is running on http://localhost:8000")

def test_additional_scenarios():
    """Test additional edge cases"""
    
    print("\n" + "="*50)
    print("Testing additional scenarios...")
    
    # Test case 2: Mixed content (identity + off-topic)
    test_request_2 = {
        "id": 103,
        "user_input": "مرحبا! كيف حالك؟",
        "new_input": [
            {
                "question": "من انت؟",
                "answer": "من انت وما هدفك؟"
            },
            {
                "question": "ما رأيك بالطقس؟",
                "answer": "الطقس جميل اليوم"
            }
        ],
        "languages": "ar"
    }
    
    print("\nTest 2: Mixed identity + off-topic")
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json=test_request_2,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Response preview:", {
                "personal_greeting_and_off_topic": result.get("personal_greeting_and_off_topic", ""),
                "description_identity": result.get("description_identity", ""),
                "status": result.get("status")
            })
            
    except Exception as e:
        print(f"Error in test 2: {e}")

if __name__ == "__main__":
    test_off_topic_scenario()
    test_additional_scenarios()
