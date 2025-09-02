#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_various_titles():
    """Test recognition of various professional titles"""
    print("🎓 Testing Various Professional Titles")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {
            "id": 1,
            "input": "مرحبا! انا المهندس احمد كيف حالك؟",
            "expected_title": "المهندس",
            "expected_name": "أحمد",
            "lang": "ar",
            "description": "Engineer (Arabic)"
        },
        {
            "id": 2,
            "input": "Hello! I'm Dr. Sarah, how are you?",
            "expected_title": "Dr.",
            "expected_name": "Sarah",
            "lang": "en",
            "description": "Doctor (English)"
        },
        {
            "id": 3,
            "input": "مرحبا! انا الدكتور محمد",
            "expected_title": "الدكتور",
            "expected_name": "محمد",
            "lang": "ar",
            "description": "Doctor (Arabic)"
        },
        {
            "id": 4,
            "input": "Hi, I'm Professor Johnson",
            "expected_title": "Professor",
            "expected_name": "Johnson",
            "lang": "en",
            "description": "Professor (English)"
        },
        {
            "id": 5,
            "input": "أهلا! انا الأستاذ علي",
            "expected_title": "الأستاذ",
            "expected_name": "علي",
            "lang": "ar",
            "description": "Professor (Arabic)"
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
        
        # Check for title and name recognition
        expected_title = test_case['expected_title']
        expected_name = test_case['expected_name']
        
        has_title = expected_title in greeting_response
        has_name = expected_name in greeting_response
        
        if has_title and has_name:
            print(f"✅ PASS: Both '{expected_title}' and '{expected_name}' found")
        elif has_name:
            print(f"🟡 PARTIAL: Name '{expected_name}' found, but title '{expected_title}' missing")
            all_passed = False
        else:
            print(f"❌ FAIL: Neither title nor name properly recognized")
            all_passed = False
        
        print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Professional title recognition working perfectly!")
    else:
        print("⚠️  SOME TESTS NEED IMPROVEMENT: Title recognition needs enhancement")
    
    return all_passed

if __name__ == "__main__":
    test_various_titles()
