#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_name_recognition():
    """Test that the AI recognizes and uses names in greetings"""
    print("🧪 Testing Name Recognition in Greetings")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {
            "id": 1,
            "input": "مرحبا! انا احمد كيف حالك؟",
            "expected_name": "أحمد",
            "lang": "ar",
            "description": "Arabic - انا احمد"
        },
        {
            "id": 2,
            "input": "Hello! I'm John, how are you?",
            "expected_name": "John",
            "lang": "en",
            "description": "English - I'm John"
        },
        {
            "id": 3,
            "input": "اسمي سارة، مرحبا بك",
            "expected_name": "سارة",
            "lang": "ar", 
            "description": "Arabic - اسمي سارة"
        },
        {
            "id": 4,
            "input": "Hi, my name is Sarah!",
            "expected_name": "Sarah",
            "lang": "en",
            "description": "English - my name is Sarah"
        },
        {
            "id": 5,
            "input": "Hello! How are you?",
            "expected_name": None,
            "lang": "en",
            "description": "No name introduction"
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print("-" * 40)
        
        result = analyzer.analyze(
            id=test_case['id'],
            user_input=test_case['input'],
            new_input=[],
            languages=test_case['lang']
        )
        
        greeting_response = result['personal_greeting_and_off_topic']
        print(f"Response: {greeting_response}")
        
        # Check if name is used when expected
        expected_name = test_case['expected_name']
        if expected_name:
            if expected_name in greeting_response:
                print(f"✅ PASS: Name '{expected_name}' found in response")
            else:
                print(f"❌ FAIL: Expected name '{expected_name}' not found")
                all_passed = False
        else:
            print("✅ PASS: No name expected (generic greeting)")
        
        print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Name recognition working perfectly!")
    else:
        print("❌ SOME TESTS FAILED: Name recognition needs improvement")
    
    return all_passed

if __name__ == "__main__":
    test_name_recognition()
