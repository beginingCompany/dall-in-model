#!/usr/bin/env python3
"""
Test GPT-based identity detection system
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_gpt_identity_detection():
    """Test the new GPT-based identity detection system"""
    
    print("🤖 TESTING GPT-BASED IDENTITY DETECTION")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test cases with variations and "trash input"
    test_cases = [
        # User's example
        ("who is ur developer", "en", "developer"),
        
        # Variations with informal language
        ("who r u", "en", "who_are_you"),
        ("wat is begining", "en", "what_is_begining"),
        ("whos ur team", "en", "team"),
        ("wat do u do", "en", "role"),
        ("who made u", "en", "developer"),
        ("y were u created", "en", "purpose"),
        ("can u understand me", "en", "understand_personality"),
        ("how u work", "en", "how_analyze"),
        ("whats ur objective", "en", "objectives"),
        
        # Trash/noise input that should still work
        ("umm who are you exactly", "en", "who_are_you"),
        ("hey, who is your developer tho?", "en", "developer"),
        ("so like what is begining", "en", "what_is_begining"),
        ("i wonder what do you actually do", "en", "role"),
        
        # Arabic variations
        ("مين انت بالضبط", "ar", "who_are_you"),
        ("من هو المطور تبعك", "ar", "developer"),
        
        # Non-identity questions (should return False)
        ("I am happy today", "en", None),
        ("How do you feel about emotions", "en", None),
        ("Tell me about your day", "en", None),
        ("What is your favorite color", "en", None),
    ]
    
    print(f"Testing {len(test_cases)} cases...")
    print("\n" + "-" * 60)
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, (question, language, expected_category) in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i:2d}: '{question}'")
        
        # Test detection
        is_detected, detected_category, response_data = analyzer.detect_identity_question(question)
        
        # Check if result matches expectation
        if expected_category is None:
            # Should NOT be detected as identity
            if not is_detected:
                print(f"✅ PASS: Correctly identified as non-identity question")
                success_count += 1
            else:
                print(f"❌ FAIL: Should NOT be identity, but detected as '{detected_category}'")
        else:
            # Should be detected as identity with correct category
            if is_detected and detected_category == expected_category:
                print(f"✅ PASS: Correctly detected as '{detected_category}'")
                success_count += 1
                
                # Test the full pipeline
                test_data = {
                    "id": 12345,
                    "user_input": question,
                    "new_input": [],
                    "languages": language
                }
                
                result = analyzer.analyze(**test_data)
                response = json.loads(result["content"])
                
                if response.get('status') == 'identity':
                    print(f"    ✅ Full pipeline: Status 'identity' returned")
                    identity_text = response.get('description_identity', '')
                    if len(identity_text) > 0:
                        print(f"    ✅ Identity response: {identity_text[:50]}...")
                    else:
                        print(f"    ❌ No identity response text")
                else:
                    print(f"    ❌ Full pipeline failed: Status '{response.get('status')}'")
                    
            else:
                if not is_detected:
                    print(f"❌ FAIL: Should be detected as '{expected_category}', but not detected")
                else:
                    print(f"❌ FAIL: Expected '{expected_category}', but got '{detected_category}'")
    
    # Summary
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "🎯" * 20)
    print("RESULTS SUMMARY")
    print("🎯" * 20)
    
    print(f"✅ Successful tests: {success_count}")
    print(f"❌ Failed tests: {total_tests - success_count}")
    print(f"📊 Total tests: {total_tests}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    # Test the specific user example
    print(f"\n🎯 SPECIFIC USER EXAMPLE TEST:")
    print("-" * 40)
    
    user_example = {
        "id": 225985882206,
        "user_input": "who is ur developer",
        "new_input": [],
        "languages": "en"
    }
    
    result = analyzer.analyze(**user_example)
    response = json.loads(result["content"])
    
    print(f"Input: 'who is ur developer'")
    print(f"Status: {response.get('status')}")
    print(f"Identity Response: {response.get('description_identity', '')}")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    print(f"Clarification Questions: {len(response.get('clarification_questions', []))}")
    
    expected_developer_response = "I was developed by a team of researchers and engineers from Saudi Arabia"
    actual_response = response.get('description_identity', '')
    
    if response.get('status') == 'identity' and "researcher" in actual_response.lower():
        print(f"✅ SUCCESS: User example now works correctly!")
    else:
        print(f"❌ ISSUE: User example still not working as expected")
    
    print(f"\n{'🎉' if success_rate >= 90 else '🔧'} {'SYSTEM READY!' if success_rate >= 90 else 'NEEDS ADJUSTMENT'}")
    
    return success_rate >= 90

if __name__ == "__main__":
    success = test_gpt_identity_detection()
    
    if success:
        print("\n✅ GPT-based identity detection working correctly!")
        print("🚀 The system can now handle variations and informal language!")
    else:
        print(f"\n❌ Some issues found. Check the details above.")
        print("🔧 Consider adjusting the GPT classification prompt.")
