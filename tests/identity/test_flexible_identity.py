#!/usr/bin/env python3
"""
Test the enhanced flexible identity detection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_flexible_identity_detection():
    print("Testing flexible identity detection...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Test various ways users might ask identity questions
    test_cases = [
        # Arabic variations
        ("من انت", "who_are_you"),
        ("اخبرني عن نفسك", "who_are_you"),
        ("عرف نفسك", "who_are_you"),
        ("ايش انت", "who_are_you"),
        ("شنو انت", "who_are_you"),
        ("ما هو مشروع BEGINING", "what_is_begining"),
        ("اشرح لي المشروع", "what_is_begining"),
        ("ايش هذا النظام", "what_is_begining"),
        ("ما هدفك", "purpose"),
        ("ليش موجود", "purpose"),
        ("ايش دورك", "role"),
        ("من طورك", "developer"),
        ("من صنعك", "developer"),
        ("كيف تشتغل", "how_analyze"),
        ("كيف تحلل الشخصية", "how_analyze"),
        
        # English variations
        ("who are you", "who_are_you"),
        ("what are you", "who_are_you"),
        ("tell me about yourself", "who_are_you"),
        ("introduce yourself", "who_are_you"),
        ("what is BEGINING", "what_is_begining"),
        ("explain this system", "what_is_begining"),
        ("about this project", "what_is_begining"),
        ("what's your purpose", "purpose"),
        ("why do you exist", "purpose"),
        ("what do you do", "purpose"),
        ("what's your role", "role"),
        ("how do you help", "role"),
        ("who made you", "developer"),
        ("who created you", "developer"),
        ("who developed you", "developer"),
        ("how do you work", "how_analyze"),
        ("how do you analyze personality", "how_analyze"),
        ("what's your method", "how_analyze"),
        
        # Non-identity (should return empty)
        ("أنا مطور", ""),  # "I am a developer" - about user, not system
        ("I work in data analysis", ""),  # About user
        ("أحب العمل مع الفرق", ""),  # "I like working with teams"
        ("I enjoy problem solving", ""),  # About user
        ("مرحبا كيف حالك", ""),  # Greeting
        ("hello how are you", ""),  # Greeting about user
    ]
    
    print("=" * 80)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (input_text, expected_category) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{input_text}'")
        
        # Test the identity detection
        response = analyzer.get_identity_response(input_text, "ar" if any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in input_text) else "en")
        
        # Check if response matches expectation
        if expected_category == "":
            # Should return empty string
            is_correct = response == ""
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"Expected: No response, Got: {'No response' if response == '' else 'Response given'}")
            print(f"Status: {status}")
        else:
            # Should return a response containing the expected content
            is_correct = response != "" and expected_category in analyzer.IDENTITY_RESPONSES
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"Expected: {expected_category} response")
            print(f"Got: {'Response given' if response else 'No response'}")
            if response:
                print(f"Response preview: {response[:100]}...")
            print(f"Status: {status}")
        
        if is_correct:
            correct_predictions += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {correct_predictions}/{total_tests} correct ({(correct_predictions/total_tests)*100:.1f}%)")
    
    if correct_predictions == total_tests:
        print("🎉 Perfect! All identity detection tests passed!")
    elif correct_predictions >= total_tests * 0.8:
        print("✅ Good! Most tests passed.")
    else:
        print("⚠️ Needs improvement. Several tests failed.")
    
    return correct_predictions, total_tests

def test_mixed_content_with_flexible_detection():
    print("\n" + "=" * 80)
    print("Testing mixed content with flexible identity detection...")
    
    analyzer = PersonalityAnalyzer()
    
    # Test mixed content scenarios
    test_scenarios = [
        {
            "input": "مرحبًا! أنا شخص يستمتع بالعمل مع البيانات. من انت؟",
            "description": "Personality description + identity question"
        },
        {
            "input": "I love problem solving and working with teams. What is your purpose?",
            "description": "English personality + identity question"
        },
        {
            "input": "أحب التحليل والإبداع. اشرح لي النظام.",
            "description": "Arabic personality + system question"
        },
        {
            "input": "من انت وما هو مشروع BEGINING؟",
            "description": "Multiple identity questions"
        },
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}: {scenario['description']}")
        print(f"Input: {scenario['input']}")
        
        # Test content separation
        cleaned = analyzer._extract_personality_content_from_mixed_input(scenario['input'])
        print(f"Cleaned personality content: {cleaned}")
        
        # Test identity detection
        identity_response = analyzer.get_identity_response(scenario['input'])
        print(f"Identity response given: {'Yes' if identity_response else 'No'}")
        
        if identity_response:
            print(f"Identity response preview: {identity_response[:100]}...")
        
        print("-" * 40)

if __name__ == "__main__":
    # Test flexible identity detection
    correct, total = test_flexible_identity_detection()
    
    # Test mixed content handling
    test_mixed_content_with_flexible_detection()
    
    print(f"\n🎯 Final Summary: {correct}/{total} identity detection tests passed")
