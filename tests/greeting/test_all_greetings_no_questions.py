#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_multiple_greeting_scenarios():
    """Test various greeting scenarios to ensure no questions are asked"""
    print("🧪 Testing Multiple Greeting Scenarios (No Questions)")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {
            "id": 1,
            "input": "Hello! How are you?",
            "lang": "en",
            "description": "English greeting"
        },
        {
            "id": 2,
            "input": "مرحبا! كيف حالك؟",
            "lang": "ar", 
            "description": "Arabic greeting"
        },
        {
            "id": 3,
            "input": "Good morning! Nice weather today.",
            "lang": "en",
            "description": "English greeting with off-topic"
        },
        {
            "id": 4,
            "input": "أهلا وسهلا! تشرفنا بلقائك",
            "lang": "ar",
            "description": "Arabic welcoming greeting"
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
        
        # Check for question indicators
        contains_question = False
        question_indicators = ['?', '؟', 'كيف يمكنني', 'How can I help', 'What can I do']
        
        for indicator in question_indicators:
            if indicator in greeting_response:
                contains_question = True
                print(f"❌ FAIL: Found question indicator: '{indicator}'")
                all_passed = False
                break
        
        if not contains_question:
            print("✅ PASS: No questions found")
        
        print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED: No questions found in greeting responses!")
    else:
        print("❌ SOME TESTS FAILED: Questions still present in responses")
    
    return all_passed

if __name__ == "__main__":
    test_multiple_greeting_scenarios()
