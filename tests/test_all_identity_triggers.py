#!/usr/bin/env python3
"""
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
