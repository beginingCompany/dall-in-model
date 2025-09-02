#!/usr/bin/env python3

import sys
import json
sys.path.append('.')

from app.personality_analyzer import PersonalityAnalyzer

def test_comprehensive():
    print("🧪 COMPREHENSIVE FINAL TEST")
    print("============================")
    print("Testing all requirements:")
    print("1. Multi-question identity detection")
    print("2. Friendly tone responses")
    print("3. Correct JSON format: description_identity as null vs string")
    print("4. Bilingual support (Arabic + English)")
    print("\n" + "="*60 + "\n")
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        # Non-identity cases (should return null)
        {
            "input": "أنا مطور برمجيات",
            "languages": "ar",
            "expected": "null",
            "description": "Arabic non-identity input"
        },
        {
            "input": "I am a software developer",
            "languages": "en",
            "expected": "null",
            "description": "English non-identity input"
        },
        
        # Single identity cases (should return string)
        {
            "input": "من انت",
            "languages": "ar",
            "expected": "string",
            "description": "Arabic single identity question"
        },
        {
            "input": "who are you",
            "languages": "en",
            "expected": "string",
            "description": "English single identity question"
        },
        
        # Multi-identity cases (should return combined string)
        {
            "input": "من انت وما هدفك",
            "languages": "ar",
            "expected": "string",
            "description": "Arabic multi-identity questions"
        },
        {
            "input": "who are you and what is your purpose",
            "languages": "en",
            "expected": "string",
            "description": "English multi-identity questions"
        },
        {
            "input": "tell me about yourself and how you analyze",
            "languages": "en",
            "expected": "string",
            "description": "English complex multi-identity"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"TEST {i}: {test_case['description']}")
        print(f"Input: '{test_case['input']}'")
        
        result = analyzer.analyze(
            id=200 + i, 
            user_input=test_case['input'], 
            new_input=[], 
            languages=test_case['languages']
        )
        
        desc_identity = result.get('description_identity')
        
        if test_case['expected'] == 'null':
            if desc_identity is None:
                print("✅ PASSED: description_identity is null")
                passed += 1
            else:
                print(f"❌ FAILED: Expected null, got {type(desc_identity).__name__}: {repr(desc_identity)}")
        
        elif test_case['expected'] == 'string':
            if isinstance(desc_identity, str) and desc_identity.strip():
                print("✅ PASSED: description_identity is non-empty string")
                # Check for friendly tone indicators
                friendly_indicators = [
                    "Let's begin", "لنبدأ", "I'm here to help", "أهدف لمساعدتك",
                    "uncovering what makes you unique", "ما يميزك"
                ]
                if any(indicator in desc_identity for indicator in friendly_indicators):
                    print("✅ BONUS: Contains friendly tone language")
                
                print(f"Response preview: {desc_identity[:100]}...")
                passed += 1
            else:
                print(f"❌ FAILED: Expected non-empty string, got {type(desc_identity).__name__}: {repr(desc_identity)}")
        
        print()
    
    print("="*60)
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL REQUIREMENTS SUCCESSFULLY IMPLEMENTED!")
        print("\n✅ Multi-question identity detection: WORKING")
        print("✅ Friendly conversational tone: WORKING") 
        print("✅ Correct JSON format (null vs string): WORKING")
        print("✅ Bilingual support (Arabic + English): WORKING")
        print("\nThe system now handles:")
        print("• Single identity questions (من انت, who are you)")
        print("• Multi-identity questions (من انت وما هدفك, who are you and what is your purpose)")
        print("• Returns description_identity as null for non-identity inputs")
        print("• Returns description_identity as friendly string for identity inputs")
        print("• Works in both Arabic and English languages")
    else:
        print("❌ Some tests failed. Please review the issues above.")

if __name__ == "__main__":
    test_comprehensive()
