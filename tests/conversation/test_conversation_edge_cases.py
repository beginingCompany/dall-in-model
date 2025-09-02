#!/usr/bin/env python3
"""
Comprehensive test for conversation history edge cases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_conversation_edge_cases():
    """Test various edge cases for conversation history processing"""
    
    print("🔬 Testing Conversation History Edge Cases")
    print("=" * 70)
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {
            "name": "English greeting with identity in history",
            "data": {
                "id": 301,
                "user_input": "Hello! How are you?",
                "new_input": [
                    {
                        "question": "Could you tell me more about yourself?",
                        "answer": "Who are you and what is your purpose?"
                    }
                ],
                "languages": "en"
            },
            "expected": ["greeting", "identity_from_history"]
        },
        {
            "name": "Greeting with personality description in history (should NOT trigger identity)",
            "data": {
                "id": 302,
                "user_input": "Hi there!",
                "new_input": [
                    {
                        "question": "Tell me about yourself",
                        "answer": "I enjoy working with data and solving analytical problems"
                    }
                ],
                "languages": "en"
            },
            "expected": ["greeting", "no_identity"]
        },
        {
            "name": "Greeting with mixed content in history",
            "data": {
                "id": 303,
                "user_input": "مرحبا!",
                "new_input": [
                    {
                        "question": "أخبرني عن نفسك",
                        "answer": "أنا مطور برمجيات أحب التحليل، ولكن أريد أن أعرف كيف تعمل"
                    }
                ],
                "languages": "ar"
            },
            "expected": ["greeting", "identity_from_mixed"]
        },
        {
            "name": "Greeting with empty conversation history",
            "data": {
                "id": 304,
                "user_input": "Hello!",
                "new_input": [],
                "languages": "en"
            },
            "expected": ["greeting", "no_identity"]
        },
        {
            "name": "Greeting with None conversation history",
            "data": {
                "id": 305,
                "user_input": "Hi!",
                "new_input": None,
                "languages": "en"
            },
            "expected": ["greeting", "no_identity"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 50)
        
        data = test_case['data']
        expected = test_case['expected']
        
        print(f"Input: {data['user_input']}")
        if data['new_input']:
            print(f"History: {data['new_input']}")
        else:
            print(f"History: {data['new_input']}")
        
        # Run analysis
        result = analyzer.analyze(
            id=data["id"],
            user_input=data["user_input"],
            new_input=data["new_input"],
            languages=data["languages"]
        )
        
        # Check results
        has_greeting = bool(result['personal_greeting_and_off_topic'])
        has_identity = result['description_identity'] is not None
        
        print(f"Result: Greeting={'✅' if has_greeting else '❌'}, Identity={'✅' if has_identity else '❌'}")
        
        # Validate expectations
        expect_greeting = "greeting" in expected
        expect_identity = any(x in expected for x in ["identity_from_history", "identity_from_mixed"])
        expect_no_identity = "no_identity" in expected
        
        greeting_ok = has_greeting == expect_greeting
        identity_ok = (has_identity and expect_identity) or (not has_identity and expect_no_identity)
        
        if greeting_ok and identity_ok:
            print(f"Status: ✅ PASS")
        else:
            print(f"Status: ❌ FAIL")
            if not greeting_ok:
                print(f"  Greeting: Expected {expect_greeting}, got {has_greeting}")
            if not identity_ok:
                print(f"  Identity: Expected {expect_identity or expect_no_identity}, got {has_identity}")
        
        if has_identity:
            print(f"Identity response: {result['description_identity'][:100]}...")
    
    print(f"\n" + "=" * 70)
    print(f"🏁 Conversation History Edge Cases Testing Complete!")

if __name__ == "__main__":
    test_conversation_edge_cases()
