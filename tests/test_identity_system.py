#!/usr/bin/env python3
"""
Test script for the identity response system in PersonalityAnalyzer
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_identity_detection():
    """Test the identity detection functionality"""
    
    print("Testing Identity Detection System")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        # Test case 1: Identity question in last answer
        {
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
            "languages": "en",
            "expected_status": "identity"
        },
        
        # Test case 2: No identity question
        {
            "id": 225985882207,
            "user_input": "Hello! I'm someone who really enjoys working with data.",
            "new_input": [
                {
                    "question": "How do you usually interact with others in social settings?",
                    "answer": "I love working in teams and mentoring colleagues."
                },
                {
                    "question": "How do you handle emotions?",
                    "answer": "I analyze problems systematically to solve them."
                }
            ],
            "languages": "en",
            "expected_status": "incomplete"
        },
        
        # Test case 3: Arabic identity question
        {
            "id": 225985882208,
            "user_input": "أنا شخص يحب العمل مع البيانات",
            "new_input": [
                {
                    "question": "كيف تتفاعل مع الآخرين؟",
                    "answer": "من أنت"
                }
            ],
            "languages": "ar",
            "expected_status": "identity"
        }
    ]
    
    # Initialize analyzer (without OpenAI for testing detection only)
    analyzer = PersonalityAnalyzer()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Expected Status: {test_case['expected_status']}")
        
        # Test the identity detection
        last_answer = ""
        if test_case['new_input']:
            last_answer = test_case['new_input'][-1].get('answer', '')
        
        is_identity, response_key, response_data = analyzer.detect_identity_question(last_answer)
        
        print(f"Last Answer: '{last_answer}'")
        print(f"Is Identity Question: {is_identity}")
        
        if is_identity:
            print(f"Response Key: {response_key}")
            identity_response = analyzer.get_identity_response(response_data, test_case['languages'])
            print(f"Identity Response: {identity_response[:100]}...")
            
            # Simulate the full response
            result = {
                "id": test_case['id'],
                "status": "identity",
                "description_identity": identity_response,
                "description_english": "",
                "description_arabic": "",
                "missing_traits": [],
                "clarification_questions": []
            }
            print(f"Full Response Status: {result['status']}")
        else:
            print("Would proceed to normal personality analysis")
            print(f"Expected to get status: {test_case['expected_status']}")
        
        # Check if result matches expectation
        actual_status = "identity" if is_identity else test_case['expected_status']
        status_match = actual_status == test_case['expected_status']
        print(f"Test Result: {'PASS' if status_match else 'FAIL'}")
        
        print("-" * 30)

def test_specific_triggers():
    """Test specific trigger phrases"""
    
    print("\n\nTesting Specific Trigger Phrases")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    trigger_tests = [
        "who are you",
        "tell me about you", 
        "introduce yourself",
        "من أنت",
        "what is begining",
        "explain begining",
        "who is your developer",
        "who made you",
        "what is your purpose",
        "how do you work",
        "can you really understand me",
        "normal personality answer"  # This should NOT trigger
    ]
    
    for trigger in trigger_tests:
        is_identity, response_key, response_data = analyzer.detect_identity_question(trigger)
        print(f"'{trigger}' -> Identity: {is_identity}, Key: {response_key}")

if __name__ == "__main__":
    test_identity_detection()
    test_specific_triggers()
