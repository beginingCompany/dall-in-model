#!/usr/bin/env python3
"""
Comprehensive test of English and Arabic identity detection
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

async def test_comprehensive_identity():
    print("🔍 COMPREHENSIVE IDENTITY DETECTION TEST")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test cases for both English and Arabic
    test_cases = [
        # English variations
        {"input": "who are you", "expected_category": "who_are_you", "language": "English"},
        {"input": "who is ur developer", "expected_category": "developer", "language": "English"},
        {"input": "who r u", "expected_category": "who_are_you", "language": "English"},
        {"input": "what is begining", "expected_category": "what_is_begining", "language": "English"},
        {"input": "what's your purpose", "expected_category": "purpose", "language": "English"},
        {"input": "what do you do", "expected_category": "role", "language": "English"},
        
        # Arabic variations
        {"input": "من أنت", "expected_category": "who_are_you", "language": "Arabic"},
        {"input": "من مطورك", "expected_category": "developer", "language": "Arabic"},
        {"input": "ما هو مشروع بيجينينغ", "expected_category": "what_is_begining", "language": "Arabic"},
        {"input": "ما هو دورك", "expected_category": "role", "language": "Arabic"},
        {"input": "ما هو هدفك", "expected_category": "purpose", "language": "Arabic"},
        
        # Non-identity questions (should return False)
        {"input": "I am happy today", "expected_category": None, "language": "English"},
        {"input": "أنا سعيد اليوم", "expected_category": None, "language": "Arabic"},
        {"input": "How are you feeling", "expected_category": None, "language": "English"},
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case["input"]
        expected_category = test_case["expected_category"]
        language = test_case["language"]
        
        print(f"\n📝 Test {i}: {language} - '{input_text}'")
        
        try:
            is_identity, detected_category, response_data = analyzer.detect_identity_question(input_text)
            
            if expected_category is None:
                # Should NOT be detected as identity
                if not is_identity:
                    print(f"✅ PASS - Correctly identified as non-identity")
                    passed += 1
                else:
                    print(f"❌ FAIL - Should not be identity, but detected as {detected_category}")
                    failed += 1
            else:
                # Should be detected as identity with correct category
                if is_identity and detected_category == expected_category:
                    print(f"✅ PASS - Correctly detected as {detected_category}")
                    if response_data:
                        lang_key = "arabic" if language == "Arabic" else "english"
                        response_text = response_data.get(lang_key, "")[:50] + "..."
                        print(f"   📄 Response: {response_text}")
                    passed += 1
                elif is_identity:
                    print(f"❌ FAIL - Expected {expected_category}, got {detected_category}")
                    failed += 1
                else:
                    print(f"❌ FAIL - Expected {expected_category}, but not detected as identity")
                    failed += 1
                    
        except Exception as e:
            print(f"💥 ERROR - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed}/{passed+failed} ({(passed/(passed+failed)*100):.1f}%)")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Identity detection is working perfectly!")
    else:
        print("⚠️ Some tests failed. Review the results above.")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_identity())
