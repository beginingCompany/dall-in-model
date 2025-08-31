#!/usr/bin/env python3
"""
<<<<<<< HEAD
Test that ALL identity response triggers are filtered out during conversations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_all_identity_triggers_in_conversation():
    """Test that ALL identity triggers are filtered out during conversations"""
    
    print("=== Testing ALL Identity Triggers in Conversation Context ===\n")
    analyzer = PersonalityAnalyzer()
    
    # Sample conversation context
    conversation_context = [
        "Q: How do you usually interact with others in social settings?\nA: I love working in teams and taking leadership roles."
    ]
    
    # Test ALL identity triggers from IDENTITY_RESPONSES
    test_cases = [
        # who_are_you triggers
        "who are you",
        "what are you", 
        "tell me about yourself",
        "introduce yourself",
        
        # what_is_begining triggers
        "what is begining",
        "explain begining",
        "tell me about begining",
        
        # purpose triggers
        "what is your purpose",
        "why were you created",
        "why are you here",
        "what's your purpose",
        
        # role triggers
        "what is your role",
        "what do you do",
        "what's your function",
        "what's your job",
        
        # developer triggers
        "who is your developer",
        "who made you",
        "who built you",
        "who created you",
        
        # team triggers
        "who is your team",
        "who's behind you",
        "who's working with you",
        "who works with you",
        
        # understand_personality triggers
        "can you really understand personality",
        "can you analyze personality",
        "can you understand me",
        "are you able to understand",
        
        # how_analyze triggers
        "how do you work",
        "how do you analyze",
        "how does this work",
        "how do you analyze personality",
        
        # objectives triggers
        "what begining aims for",
        "what are begining objectives",
        "objectives of begining",
        "goals of begining"
    ]
    
    print(f"Testing {len(test_cases)} identity trigger phrases in conversation context...\n")
    
    passed_tests = 0
    failed_tests = 0
    
    for i, trigger in enumerate(test_cases, 1):
        print(f"Test {i}: '{trigger}'")
        
        # Test the identity response in conversation context
        identity_response = analyzer.get_identity_response(
            trigger,
            language="en",
            openai_client=None,
            conversation_context=conversation_context
        )
        
        if identity_response:
            print(f"  ❌ FAILED: Identity response triggered: {identity_response[:50]}...")
            failed_tests += 1
        else:
            print(f"  ✅ PASSED: No identity response (correctly filtered)")
            passed_tests += 1
        
        print()
    
    print("="*60)
    print("SUMMARY:")
    print(f"✅ Passed: {passed_tests}/{len(test_cases)}")
    print(f"❌ Failed: {failed_tests}/{len(test_cases)}")
    
    if failed_tests == 0:
        print("🎉 ALL IDENTITY TRIGGERS CORRECTLY FILTERED IN CONVERSATION CONTEXT!")
        print("✅ System will respond to these questions but skip trait processing")
    else:
        print("⚠️ Some identity triggers are still getting through the filter")
    
    return failed_tests == 0

def test_identity_triggers_standalone():
    """Test that identity triggers still work when asked as standalone questions"""
    
    print("\n=== Testing Identity Triggers as Standalone Questions ===\n")
    analyzer = PersonalityAnalyzer()
    
    # Test a few key triggers without conversation context
    standalone_tests = [
        "who are you",
        "what is begining", 
        "what is your purpose",
        "who is your developer"
    ]
    
    passed_tests = 0
    
    for trigger in standalone_tests:
        print(f"Testing standalone: '{trigger}'")
        
        identity_response = analyzer.get_identity_response(
            trigger,
            language="en",
            openai_client=None,
            conversation_context=None  # No conversation context
        )
        
        if identity_response:
            print(f"  ✅ PASSED: Identity response provided: {identity_response[:50]}...")
            passed_tests += 1
        else:
            print(f"  ❌ FAILED: No identity response for standalone question")
        
        print()
    
    print(f"Standalone tests: {passed_tests}/{len(standalone_tests)} passed")
    return passed_tests == len(standalone_tests)

if __name__ == "__main__":
    print("Testing comprehensive identity trigger filtering...\n")
    
    test1_passed = test_all_identity_triggers_in_conversation()
    test2_passed = test_identity_triggers_standalone()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print(f"Conversation filtering: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Standalone detection: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 ALL IDENTITY RESPONSE FILTERING IS WORKING PERFECTLY!")
        print("✅ Confused answers filtered out in conversations")
        print("✅ Genuine questions answered when standalone")
    else:
        print("⚠️ Some issues remain with identity response filtering")
=======
Comprehensive test for all identity triggers and categories
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_all_identity_triggers():
    """Test all identity triggers across all 9 categories"""
    
    print("🔍 COMPREHENSIVE IDENTITY TRIGGER TEST")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Get all identity responses for testing
    identity_responses = PersonalityAnalyzer.IDENTITY_RESPONSES
    
    print(f"Testing {len(identity_responses)} identity categories:")
    for category in identity_responses.keys():
        print(f"  - {category}")
    
    print("\n" + "=" * 60)
    
    # Test results tracking
    total_triggers = 0
    working_triggers = 0
    failed_triggers = []
    
    # Test each category and its triggers
    for category, response_data in identity_responses.items():
        print(f"\n🧪 TESTING CATEGORY: {category.upper()}")
        print("-" * 40)
        
        triggers = response_data.get("triggers", [])
        english_response = response_data.get("english", "")
        arabic_response = response_data.get("arabic", "")
        
        print(f"Triggers to test: {len(triggers)}")
        print(f"English response available: {len(english_response) > 0}")
        print(f"Arabic response available: {len(arabic_response) > 0}")
        
        category_working = 0
        category_total = len(triggers)
        
        for i, trigger in enumerate(triggers, 1):
            total_triggers += 1
            
            # Test detection
            is_detected, detected_key, detected_data = analyzer.detect_identity_question(trigger)
            
            # Determine expected language
            is_arabic_trigger = any('\u0600' <= char <= '\u06FF' for char in trigger)
            test_language = "ar" if is_arabic_trigger else "en"
            
            if is_detected:
                # Test standalone question
                test_data = {
                    "id": 12345,
                    "user_input": trigger,
                    "new_input": [],
                    "languages": test_language
                }
                
                result = analyzer.analyze(**test_data)
                response = json.loads(result["content"])
                
                # Verify response
                status_correct = response.get('status') == 'identity'
                has_identity_text = len(response.get('description_identity', '')) > 0
                has_questions = len(response.get('clarification_questions', [])) > 0
                
                if status_correct and has_identity_text and has_questions:
                    working_triggers += 1
                    category_working += 1
                    print(f"  ✅ {i:2d}. '{trigger}' -> WORKING")
                else:
                    failed_triggers.append({
                        'category': category,
                        'trigger': trigger,
                        'issue': f"Status: {response.get('status')}, Identity: {has_identity_text}, Questions: {has_questions}"
                    })
                    print(f"  ❌ {i:2d}. '{trigger}' -> FAILED ({response.get('status')})")
            else:
                failed_triggers.append({
                    'category': category, 
                    'trigger': trigger,
                    'issue': 'Not detected by detect_identity_question()'
                })
                print(f"  ❌ {i:2d}. '{trigger}' -> NOT DETECTED")
        
        print(f"Category result: {category_working}/{category_total} triggers working")
    
    # Final summary
    print("\n" + "🎯" * 20)
    print("FINAL RESULTS SUMMARY")
    print("🎯" * 20)
    
    success_rate = (working_triggers / total_triggers * 100) if total_triggers > 0 else 0
    
    print(f"✅ Working triggers: {working_triggers}")
    print(f"❌ Failed triggers: {len(failed_triggers)}")
    print(f"📊 Total triggers tested: {total_triggers}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if failed_triggers:
        print(f"\n🔧 FAILED TRIGGERS DETAILS:")
        print("-" * 50)
        for failure in failed_triggers:
            print(f"Category: {failure['category']}")
            print(f"Trigger: '{failure['trigger']}'")
            print(f"Issue: {failure['issue']}")
            print("-" * 30)
    
    # Test specific examples from each category
    print(f"\n🧪 TESTING SPECIFIC EXAMPLES:")
    print("-" * 50)
    
    test_cases = [
        ("who are you", "en", "who_are_you"),
        ("what is begining", "en", "what_is_begining"),
        ("purpose", "en", "purpose"),
        ("what do you do", "en", "role"),
        ("who made you", "en", "developer"),
        ("who is your team", "en", "team"),
        ("can you analyze me", "en", "understand_personality"),
        ("how do you work", "en", "how_analyze"),
        ("what begining aims for", "en", "objectives"),
        ("من أنت", "ar", "who_are_you"),
        ("ما هو BEGINING", "ar", "what_is_begining"),
    ]
    
    for trigger, language, expected_category in test_cases:
        test_data = {
            "id": 99999,
            "user_input": trigger,
            "new_input": [],
            "languages": language
        }
        
        result = analyzer.analyze(**test_data)
        response = json.loads(result["content"])
        
        status = response.get('status')
        identity_text = response.get('description_identity', '')
        
        if status == 'identity' and len(identity_text) > 0:
            print(f"✅ '{trigger}' ({language}) -> WORKING")
        else:
            print(f"❌ '{trigger}' ({language}) -> FAILED (status: {status})")
    
    print(f"\n{'🎉' if success_rate >= 90 else '🔧'} {'SUCCESS!' if success_rate >= 90 else 'NEEDS FIXES'}")
    
    return success_rate >= 90, failed_triggers

if __name__ == "__main__":
    success, failures = test_all_identity_triggers()
    
    if success:
        print("\n✅ All identity triggers working correctly!")
    else:
        print(f"\n❌ Some triggers need fixing. Check the details above.")
        print("🔧 Recommend investigating detection logic or trigger patterns.")
>>>>>>> f912c397f4608be37933b416c471652681384d61
