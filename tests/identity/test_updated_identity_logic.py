#!/usr/bin/env python3
"""
Test the updated identity logic:
- Identity question in the middle (answered already) → should NOT return identity
- Identity question as last answer → should return identity
- Standalone identity question → should return identity
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_updated_logic():
    """Test the updated identity logic"""
    
    print("🧪 TESTING UPDATED IDENTITY LOGIC")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test Case 1: Identity question already answered (in middle) - should NOT return identity
    print("TEST 1: Identity Question Already Answered (Middle) - Should NOT return identity")
    print("-" * 60)
    
    case1 = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles."
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "i analytical can solving the problems by analyze them"
            }
        ],
        "languages": "en"
    }
    
    result1 = analyzer.analyze(**case1)
    response1 = json.loads(result1["content"])
    
    print(f"Status: {response1.get('status')}")
    print(f"Identity Response: {response1.get('description_identity')}")
    print(f"Expected: Status should be 'incomplete', NOT 'identity'")
    
    expected_status_1 = response1.get('status') != 'identity'
    print(f"✅ PASS: {expected_status_1}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Test Case 2: Identity question as LAST answer - should return identity
    print("TEST 2: Identity Question as Last Answer - Should return identity")
    print("-" * 60)
    
    case2 = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles."
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            }
        ],
        "languages": "en"
    }
    
    result2 = analyzer.analyze(**case2)
    response2 = json.loads(result2["content"])
    
    print(f"Status: {response2.get('status')}")
    print(f"Identity Response: {response2.get('description_identity', '')[:50]}...")
    print(f"Expected: Status should be 'identity'")
    
    expected_status_2 = response2.get('status') == 'identity'
    print(f"✅ PASS: {expected_status_2}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Test Case 3: Standalone identity question - should return identity
    print("TEST 3: Standalone Identity Question - Should return identity")
    print("-" * 60)
    
    case3 = {
        "id": 123456789,
        "user_input": "who are you",
        "new_input": [],
        "languages": "en"
    }
    
    result3 = analyzer.analyze(**case3)
    response3 = json.loads(result3["content"])
    
    print(f"Status: {response3.get('status')}")
    print(f"Identity Response: {response3.get('description_identity', '')[:50]}...")
    print(f"Expected: Status should be 'identity'")
    
    expected_status_3 = response3.get('status') == 'identity'
    print(f"✅ PASS: {expected_status_3}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Test Case 4: Arabic standalone identity question - should return identity
    print("TEST 4: Arabic Standalone Identity Question - Should return identity")
    print("-" * 60)
    
    case4 = {
        "id": 65387652876,
        "user_input": "من انت",
        "new_input": [],
        "languages": "ar"
    }
    
    result4 = analyzer.analyze(**case4)
    response4 = json.loads(result4["content"])
    
    print(f"Status: {response4.get('status')}")
    print(f"Identity Response (Arabic): {response4.get('description_identity', '')[:50]}...")
    print(f"Expected: Status should be 'identity'")
    
    expected_status_4 = response4.get('status') == 'identity'
    print(f"✅ PASS: {expected_status_4}")
    
    # Summary
    print("\n" + "🎯" * 20)
    print("SUMMARY OF TESTS:")
    print("🎯" * 20)
    
    all_pass = all([expected_status_1, expected_status_2, expected_status_3, expected_status_4])
    
    print(f"✅ Test 1 (Identity in middle - should NOT return identity): {'PASS' if expected_status_1 else 'FAIL'}")
    print(f"✅ Test 2 (Identity as last answer - should return identity): {'PASS' if expected_status_2 else 'FAIL'}")
    print(f"✅ Test 3 (Standalone English identity - should return identity): {'PASS' if expected_status_3 else 'FAIL'}")
    print(f"✅ Test 4 (Standalone Arabic identity - should return identity): {'PASS' if expected_status_4 else 'FAIL'}")
    
    if all_pass:
        print("\n🎉 ALL TESTS PASSED! Updated logic is working correctly!")
        print("💡 Identity responses are only given for CURRENT identity questions, not past ones.")
    else:
        print("\n❌ Some tests failed. Need to check the logic.")
    
    return all_pass

if __name__ == "__main__":
    test_updated_logic()
