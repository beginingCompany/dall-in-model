#!/usr/bin/env python3
"""
Test complete identity detection system with informal variations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_complete_informal_detection():
    print("🔍 TESTING COMPLETE IDENTITY DETECTION WITH INFORMAL VARIATIONS")
    print("=" * 70)
    
    analyzer = PersonalityAnalyzer()
    
    # Test cases with very informal variations that users might type
    test_cases = [
        # Informal purpose questions
        {"input": "what ur purpose", "expected": "purpose"},
        {"input": "what's ur goal", "expected": "purpose"},
        {"input": "why u here", "expected": "purpose"},
        {"input": "ur mission", "expected": "purpose"},
        
        # Informal role questions  
        {"input": "what u do", "expected": "role"},
        {"input": "ur job", "expected": "role"},
        {"input": "what's ur function", "expected": "role"},
        
        # Informal developer questions
        {"input": "who ur developer", "expected": "developer"},
        {"input": "ur creator", "expected": "developer"},
        {"input": "who made u", "expected": "developer"},
        
        # Informal identity questions
        {"input": "who u", "expected": "who_are_you"},
        {"input": "ur identity", "expected": "who_are_you"},
        
        # Arabic informal
        {"input": "هدفك", "expected": "purpose"},
        {"input": "مطورك", "expected": "developer"},
        
        # Mixed case and punctuation
        {"input": "What's UR purpose?", "expected": "purpose"},
        {"input": "WHO UR DEVELOPER", "expected": "developer"},
        
        # Should not match
        {"input": "I like programming", "expected": None},
        {"input": "how are you today", "expected": None},
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case["input"]
        expected_category = test_case["expected"]
        
        print(f"\n📝 Test {i}: '{input_text}'")
        
        try:
            # Test the complete detection system (GPT + fallback)
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
                        # Show a preview of the response
                        eng_response = response_data.get("english", "")[:50] + "..."
                        print(f"   📄 Response preview: {eng_response}")
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
    
    print(f"\n{'='*70}")
    print(f"📊 COMPLETE INFORMAL DETECTION RESULTS:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed}/{passed+failed} ({(passed/(passed+failed)*100):.1f}%)")
    
    if failed == 0:
        print("🎉 PERFECT! The model can now predict and match informal keywords perfectly!")
        print("💬 Users can now type things like 'what ur purpose' and get correct responses!")
    else:
        print("⚠️ Some tests failed. Review the results above.")

if __name__ == "__main__":
    test_complete_informal_detection()
