#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_full_analyze_method():
    """Test the complete analyze method with enhanced identity detection"""
    
    print("=== Testing Complete Analyze Method ===\n")
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        # English identity questions
        {
            "id": 12345,
            "user_input": "Who are you?",
            "languages": "en",
            "description": "English identity question"
        },
        {
            "id": 12346,
            "user_input": "Who made you?",
            "languages": "en",
            "description": "English developer question"
        },
        {
            "id": 12347,
            "user_input": "What is your purpose?",
            "languages": "en",
            "description": "English purpose question"
        },
        
        # Arabic identity questions
        {
            "id": 12348,
            "user_input": "من أنت؟",
            "languages": "ar",
            "description": "Arabic identity question"
        },
        {
            "id": 12349,
            "user_input": "من طورك؟",
            "languages": "ar",
            "description": "Arabic developer question"
        },
        {
            "id": 12350,
            "user_input": "ما هو هدفك؟",
            "languages": "ar",
            "description": "Arabic purpose question"
        },
        
        # Variations
        {
            "id": 12351,
            "user_input": "Tell me about yourself",
            "languages": "en",
            "description": "English self-introduction"
        },
        {
            "id": 12352,
            "user_input": "عرفني على نفسك",
            "languages": "ar",
            "description": "Arabic self-introduction"
        },
        
        # Language auto-detection tests
        {
            "id": 12353,
            "user_input": "Who are you?",  # English input
            "languages": "ar",  # Arabic requested
            "description": "English input with Arabic language preference"
        },
        {
            "id": 12354,
            "user_input": "من أنت؟",  # Arabic input
            "languages": "en",  # English requested
            "description": "Arabic input with English language preference"
        },
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{total_count}: {test_case['description']}")
        print(f"Input: '{test_case['user_input']}'")
        print(f"Requested Language: {test_case['languages']}")
        
        try:
            result = analyzer.analyze(
                id=test_case["id"],
                user_input=test_case["user_input"],
                new_input=[],
                languages=test_case["languages"]
            )
            
            # Check if description_identity is present and not null
            identity_response = result.get("description_identity")
            if identity_response:
                print(f"✅ Identity Response Found:")
                print(f"   {identity_response[:100]}...")
                
                # Determine response language
                has_arabic = any('\u0600' <= c <= '\u06FF' for c in identity_response)
                response_lang = "Arabic" if has_arabic else "English"
                print(f"   Response Language: {response_lang}")
                
                success_count += 1
            else:
                print(f"❌ No identity response found")
                print(f"   Result: {result}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 70)
    
    print(f"\n=== Results ===")
    print(f"Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # Additional test for non-identity questions
    print(f"\n=== Testing Non-Identity Questions ===")
    
    non_identity_tests = [
        {
            "id": 99991,
            "user_input": "I am a software developer who likes to work alone",
            "languages": "en",
            "description": "Normal personality input"
        },
        {
            "id": 99992,
            "user_input": "أنا مطور برمجيات أحب العمل بمفردي",
            "languages": "ar",
            "description": "Normal personality input in Arabic"
        }
    ]
    
    for test in non_identity_tests:
        print(f"Testing: {test['description']}")
        print(f"Input: '{test['user_input']}'")
        
        try:
            result = analyzer.analyze(
                id=test["id"],
                user_input=test["user_input"],
                new_input=[],
                languages=test["languages"]
            )
            
            identity_response = result.get("description_identity")
            if identity_response:
                print(f"❌ Unexpected identity response: {identity_response[:50]}...")
            else:
                print(f"✅ No identity response (correct for non-identity input)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_full_analyze_method()
