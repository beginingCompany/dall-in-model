#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_enhanced_identity_detection():
    print("=== Testing Enhanced Identity Detection ===\n")
    
    # Test cases with various phrasings in English and Arabic
    test_cases = [
        # English variations
        ("Who are you?", "en"),
        ("Tell me about yourself", "en"),
        ("Can you introduce yourself?", "en"),
        ("Who made you?", "en"),
        ("Who is your creator?", "en"),
        ("Who developed you?", "en"),
        ("What is your purpose?", "en"),
        ("Why were you created?", "en"),
        ("What do you do?", "en"),
        ("What is your role?", "en"),
        ("How do you work?", "en"),
        ("How do you analyze personality?", "en"),
        ("Can you really understand me?", "en"),
        ("Are you accurate?", "en"),
        ("What is BEGINING?", "en"),
        ("Explain BEGINING to me", "en"),
        
        # Arabic variations
        ("من أنت؟", "ar"),
        ("مين انت؟", "ar"),
        ("عرفني على نفسك", "ar"),
        ("من صنعك؟", "ar"),
        ("من طورك؟", "ar"),
        ("من هو مطورك؟", "ar"),
        ("ما هو هدفك؟", "ar"),
        ("ايش هدفك؟", "ar"),
        ("ليش انت هنا؟", "ar"),
        ("شو بتعمل؟", "ar"),
        ("ما هو دورك؟", "ar"),
        ("كيف تعمل؟", "ar"),
        ("كيف تحلل الشخصية؟", "ar"),
        ("هل يمكنك فهم شخصيتي؟", "ar"),
        ("تقدر تحللني؟", "ar"),
        ("ما هو BEGINING؟", "ar"),
        ("شرحلي BEGINING", "ar"),
        
        # Mixed and variations
        ("Who's your developer?", "en"),
        ("What's your team like?", "en"),
        ("مين عملك؟", "ar"),
        ("من يعمل معك؟", "ar"),
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for question, expected_lang in test_cases:
        print(f"Testing: '{question}' (Expected: {expected_lang})")
        
        # Test with language auto-detection
        response = PersonalityAnalyzer.get_identity_response(question, expected_lang)
        
        if response:
            print(f"✅ Response found: {response[:100]}...")
            success_count += 1
        else:
            print(f"❌ No response found")
        
        print("-" * 50)
    
    print(f"\n=== Results ===")
    print(f"Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # Test language auto-detection
    print(f"\n=== Testing Language Auto-Detection ===")
    
    auto_detect_tests = [
        ("Who are you?", "en"),  # Should detect English
        ("من أنت؟", "ar"),       # Should detect Arabic
        ("Who made you مين صنعك؟", "mixed"),  # Mixed text
    ]
    
    for question, test_type in auto_detect_tests:
        print(f"Testing auto-detection: '{question}'")
        
        # Test with default language but let auto-detection work
        response_en = PersonalityAnalyzer.get_identity_response(question, "en")
        response_ar = PersonalityAnalyzer.get_identity_response(question, "ar")
        
        print(f"EN response: {response_en[:50] if response_en else 'None'}...")
        print(f"AR response: {response_ar[:50] if response_ar else 'None'}...")
        print("-" * 50)

if __name__ == "__main__":
    test_enhanced_identity_detection()
