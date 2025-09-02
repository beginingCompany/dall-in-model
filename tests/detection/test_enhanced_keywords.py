#!/usr/bin/env python3
"""
Test enhanced keyword detection for variations like "what ur purpose"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_enhanced_keyword_detection():
    print("🔍 TESTING ENHANCED KEYWORD DETECTION")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test cases with informal variations
    test_cases = [
        # Purpose variations
        {"input": "what ur purpose", "expected": "purpose"},
        {"input": "what's ur purpose", "expected": "purpose"},
        {"input": "what is ur purpose", "expected": "purpose"},
        {"input": "ur purpose", "expected": "purpose"},
        {"input": "what's your purpose", "expected": "purpose"},
        
        # Role variations
        {"input": "what ur role", "expected": "role"},
        {"input": "ur role", "expected": "role"},
        {"input": "what u do", "expected": "role"},
        {"input": "ur job", "expected": "role"},
        
        # Developer variations
        {"input": "ur developer", "expected": "developer"},
        {"input": "who ur developer", "expected": "developer"},
        {"input": "ur creator", "expected": "developer"},
        
        # Identity variations
        {"input": "who u", "expected": "who_are_you"},
        {"input": "ur identity", "expected": "who_are_you"},
        
        # Team variations
        {"input": "ur team", "expected": "team"},
        {"input": "who behind you", "expected": "team"},
        
        # Arabic variations
        {"input": "هدفك", "expected": "purpose"},
        {"input": "دورك", "expected": "role"},
        {"input": "مطورك", "expected": "developer"},
        
        # Should NOT match
        {"input": "I am happy today", "expected": None},
        {"input": "how are you feeling", "expected": None},
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case["input"]
        expected_category = test_case["expected"]
        
        print(f"\n📝 Test {i}: '{input_text}'")
        
        try:
            # Test the fallback detection directly
            is_identity, detected_category, response_data = analyzer._fallback_identity_detection(input_text)
            
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
    print(f"📊 ENHANCED KEYWORD DETECTION RESULTS:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed}/{passed+failed} ({(passed/(passed+failed)*100):.1f}%)")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Enhanced keyword detection is working perfectly!")
    else:
        print("⚠️ Some tests failed. Review the results above.")

if __name__ == "__main__":
    test_enhanced_keyword_detection()
