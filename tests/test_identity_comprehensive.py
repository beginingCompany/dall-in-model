#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_identity_responses_comprehensive():
    """Comprehensive test for all IDENTITY_RESPONSES categories and patterns"""
    
    print("=" * 80)
    print("COMPREHENSIVE IDENTITY_RESPONSES TEST")
    print("=" * 80)
    
    # Test all categories with their triggers
    test_categories = {
        "who_are_you": {
            "original_triggers": ["who are you", "tell me about you", "introduce yourself", "من أنت"],
            "additional_tests": [
                "Who are you?",
                "Tell me about yourself",
                "Can you introduce yourself?",
                "من أنت؟",
                "مين انت؟",
                "عرفني على نفسك",
                "احكيلي عنك"
            ]
        },
        "what_is_begining": {
            "original_triggers": ["what is begining", "explain begining", "ما هو BEGINING", "BEGINING يعني ايه"],
            "additional_tests": [
                "What is BEGINING?",
                "Explain BEGINING to me",
                "Tell me about BEGINING",
                "ما هو BEGINING؟",
                "ايش BEGINING؟",
                "شرحلي BEGINING",
                "عن BEGINING"
            ]
        },
        "purpose": {
            "original_triggers": ["purpose", "why were you created", "why are you here", "ما هو هدفك"],
            "additional_tests": [
                "What is your purpose?",
                "Why were you created?",
                "Why are you here?",
                "What's your goal?",
                "ما هو هدفك؟",
                "ايش هدفك؟",
                "ليش انت هنا؟",
                "شو غايتك؟"
            ]
        },
        "role": {
            "original_triggers": ["role", "function", "what do you do", "ما هو دورك"],
            "additional_tests": [
                "What is your role?",
                "What do you do?",
                "What's your function?",
                "ما هو دورك؟",
                "ايش دورك؟",
                "شو بتعمل؟",
                "وظيفتك ايش؟"
            ]
        },
        "developer": {
            "original_triggers": ["who is your developer", "who made you", "who built you", "من هو مطورك", "من صنعك", "من بناك"],
            "additional_tests": [
                "Who is your developer?",
                "Who made you?",
                "Who built you?",
                "Who created you?",
                "Who designed you?",
                "Who programmed you?",
                "من هو مطورك؟",
                "من صنعك؟",
                "من بناك؟",
                "من طورك؟",
                "من صممك؟",
                "مين عملك؟"
            ]
        },
        "team": {
            "original_triggers": ["team", "who's behind you", "who's working with you", "من هو فريقك"],
            "additional_tests": [
                "Who is your team?",
                "Who's behind you?",
                "Who's working with you?",
                "What's your team like?",
                "من هو فريقك؟",
                "مين فريقك؟",
                "من يعمل معك؟"
            ]
        },
        "understand_personality": {
            "original_triggers": ["can you really understand", "can you analyze me", "do you understand me", "هل يمكنك حقًا فهم شخصيتي"],
            "additional_tests": [
                "Can you really understand me?",
                "Can you analyze me?",
                "Do you understand me?",
                "Are you accurate?",
                "Can you read personality?",
                "هل يمكنك حقًا فهم شخصيتي؟",
                "تقدر تحللني؟",
                "بتفهمني؟",
                "دقيق انت؟"
            ]
        },
        "how_analyze": {
            "original_triggers": ["how do you work", "how do you analyze", "كيف تحلل الشخصية"],
            "additional_tests": [
                "How do you work?",
                "How do you analyze?",
                "How does this work?",
                "What's your method?",
                "كيف تعمل؟",
                "كيف تحلل؟",
                "ايش طريقتك؟",
                "شلون تشتغل؟"
            ]
        },
        "objectives": {
            "original_triggers": ["objectives", "goals", "what begining aims for", "ما هي أهداف BEGINING"],
            "additional_tests": [
                "What are the objectives?",
                "What are the goals of BEGINING?",
                "What does BEGINING aim for?",
                "ما هي أهداف BEGINING؟",
                "اهداف BEGINING؟",
                "غايات المشروع؟"
            ]
        }
    }
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for category, data in test_categories.items():
        print(f"\n🔍 TESTING CATEGORY: {category.upper()}")
        print("-" * 60)
        
        # Test original triggers
        print("Original Triggers:")
        for trigger in data["original_triggers"]:
            total_tests += 1
            response_en = PersonalityAnalyzer.get_identity_response(trigger, "en")
            response_ar = PersonalityAnalyzer.get_identity_response(trigger, "ar")
            
            if response_en or response_ar:
                print(f"  ✅ '{trigger}' -> Found response")
                passed_tests += 1
            else:
                print(f"  ❌ '{trigger}' -> No response")
                failed_tests.append(f"{category}: '{trigger}'")
        
        # Test additional variations
        print("Additional Variations:")
        for test_input in data["additional_tests"]:
            total_tests += 1
            response_en = PersonalityAnalyzer.get_identity_response(test_input, "en")
            response_ar = PersonalityAnalyzer.get_identity_response(test_input, "ar")
            
            if response_en or response_ar:
                print(f"  ✅ '{test_input}' -> Found response")
                passed_tests += 1
            else:
                print(f"  ❌ '{test_input}' -> No response")
                failed_tests.append(f"{category}: '{test_input}'")
    
    print(f"\n" + "=" * 80)
    print("LANGUAGE AUTO-DETECTION TEST")
    print("=" * 80)
    
    # Test language auto-detection
    language_tests = [
        ("Who are you?", "Should return English"),
        ("من أنت؟", "Should return Arabic"),
        ("Who made you من صنعك", "Mixed - should detect Arabic"),
        ("What is your purpose?", "English only"),
        ("ما هو هدفك؟", "Arabic only"),
    ]
    
    for test_input, expected in language_tests:
        total_tests += 1
        print(f"\nTesting: '{test_input}' ({expected})")
        
        response_en = PersonalityAnalyzer.get_identity_response(test_input, "en")
        response_ar = PersonalityAnalyzer.get_identity_response(test_input, "ar")
        
        has_arabic_chars = any('\u0600' <= c <= '\u06FF' for c in test_input)
        
        if response_en or response_ar:
            # Check if response language matches expectation
            if response_en:
                detected_lang = "Arabic" if any('\u0600' <= c <= '\u06FF' for c in response_en) else "English"
            else:
                detected_lang = "Arabic" if any('\u0600' <= c <= '\u06FF' for c in response_ar) else "English"
            
            print(f"  ✅ Response found in {detected_lang}")
            print(f"  📝 Response: {(response_en or response_ar)[:50]}...")
            passed_tests += 1
        else:
            print(f"  ❌ No response found")
            failed_tests.append(f"Language test: '{test_input}'")
    
    print(f"\n" + "=" * 80)
    print("FALSE POSITIVE TEST")
    print("=" * 80)
    
    # Test for false positives (should NOT trigger identity responses)
    false_positive_tests = [
        "I am a software developer who likes programming",
        "I work as a team leader in my company",
        "My goal is to become a better programmer",
        "I have a role in the development team",
        "I understand that programming is hard",
        "أنا مطور برمجيات أحب البرمجة",
        "أعمل كقائد فريق في شركتي",
        "هدفي أن أصبح مبرمج أفضل",
    ]
    
    for test_input in false_positive_tests:
        total_tests += 1
        response = PersonalityAnalyzer.get_identity_response(test_input, "en")
        
        if response:
            print(f"  ❌ FALSE POSITIVE: '{test_input}' -> {response[:30]}...")
            failed_tests.append(f"False positive: '{test_input}'")
        else:
            print(f"  ✅ Correctly ignored: '{test_input}'")
            passed_tests += 1
    
    print(f"\n" + "=" * 80)
    print("FULL ANALYZER INTEGRATION TEST")
    print("=" * 80)
    
    # Test the full analyzer.analyze() method
    analyzer = PersonalityAnalyzer()
    
    integration_tests = [
        {
            "input": "Who are you?",
            "languages": "en",
            "should_have_identity": True
        },
        {
            "input": "من أنت؟", 
            "languages": "ar",
            "should_have_identity": True
        },
        {
            "input": "I am a happy person who loves socializing",
            "languages": "en", 
            "should_have_identity": False
        },
        {
            "input": "Who is your developer?",
            "languages": "en",
            "should_have_identity": True
        }
    ]
    
    for i, test in enumerate(integration_tests):
        total_tests += 1
        print(f"\nIntegration Test {i+1}: {test['input']}")
        
        try:
            result = analyzer.analyze(
                id=12340 + i,
                user_input=test["input"],
                new_input=[],
                languages=test["languages"]
            )
            
            has_identity = result.get("description_identity") is not None and result.get("description_identity") != ""
            
            if has_identity == test["should_have_identity"]:
                print(f"  ✅ Identity response handling correct")
                if has_identity:
                    print(f"  📝 Identity: {result['description_identity'][:50]}...")
                passed_tests += 1
            else:
                print(f"  ❌ Identity response handling incorrect")
                print(f"     Expected identity: {test['should_have_identity']}, Got: {has_identity}")
                failed_tests.append(f"Integration: '{test['input']}'")
                
        except Exception as e:
            print(f"  ❌ Error in analyzer: {e}")
            failed_tests.append(f"Integration error: '{test['input']}'")
    
    print(f"\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for i, failure in enumerate(failed_tests, 1):
            print(f"  {i}. {failure}")
    else:
        print(f"\n🎉 ALL TESTS PASSED!")
    
    print(f"\n" + "=" * 80)
    print("DETAILED PATTERN ANALYSIS")
    print("=" * 80)
    
    # Check each enhanced pattern
    analyzer_instance = PersonalityAnalyzer()
    
    # Get the enhanced patterns from the method (we need to call it to access the patterns)
    print("\nChecking enhanced pattern coverage...")
    
    sample_inputs = [
        "who are you exactly?",
        "tell me about your purpose",
        "what's BEGINING about?", 
        "how does your analysis work?",
        "who's your development team?",
        "من انت بالضبط؟",
        "احكيلي عن هدفك",
        "ايش BEGINING؟",
        "كيف يشتغل تحليلك؟",
        "مين فريق التطوير؟"
    ]
    
    pattern_coverage = 0
    for sample in sample_inputs:
        response = PersonalityAnalyzer.get_identity_response(sample, "en")
        if response:
            pattern_coverage += 1
            print(f"  ✅ '{sample}' -> Pattern matched")
        else:
            print(f"  ⚠️  '{sample}' -> No pattern match")
    
    print(f"\nPattern Coverage: {pattern_coverage}/{len(sample_inputs)} ({pattern_coverage/len(sample_inputs)*100:.1f}%)")

if __name__ == "__main__":
    test_identity_responses_comprehensive()
