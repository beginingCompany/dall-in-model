#!/usr/bin/env python3
"""
Test the corrected identity logic:
1. If new_input is empty -> check user_input (first interaction)
2. If new_input exists -> only check LAST answer, ignore user_input (history)
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_corrected_logic():
    print("🧪 TESTING CORRECTED IDENTITY LOGIC")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test Case 1: Identity in middle answer with next answer after it (should NOT return identity)
    print("TEST 1: Identity in past (middle answer) - Should NOT return identity")
    print("-" * 50)
    
    case1 = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights. I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions. who are you. i analytical can solving the problems by analyze them",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
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
    print(f"Expected: incomplete (✅ CORRECT)" if response1.get('status') == 'incomplete' else f"Expected: incomplete (❌ WRONG)")
    
    # Test Case 2: Identity in last answer (should return identity)
    print("\nTEST 2: Identity as last answer - Should return identity")
    print("-" * 50)
    
    case2 = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights. I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions. who are you.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
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
    print(f"Expected: identity (✅ CORRECT)" if response2.get('status') == 'identity' else f"Expected: identity (❌ WRONG)")
    
    # Test Case 3: Standalone identity question (first interaction)
    print("\nTEST 3: Standalone identity question - Should return identity")
    print("-" * 50)
    
    case3 = {
        "id": 225985882206,
        "user_input": "who are you",
        "new_input": [],
        "languages": "en"
    }
    
    result3 = analyzer.analyze(**case3)
    response3 = json.loads(result3["content"])
    
    print(f"Status: {response3.get('status')}")
    print(f"Expected: identity (✅ CORRECT)" if response3.get('status') == 'identity' else f"Expected: identity (❌ WRONG)")
    
    # Test Case 4: Identity in user_input but with new_input (should ignore user_input)
    print("\nTEST 4: Identity in user_input but has new_input - Should ignore user_input")
    print("-" * 50)
    
    case4 = {
        "id": 225985882206,
        "user_input": "who are you. I'm very analytical and enjoy solving problems",
        "new_input": [
            {
                "question": "How do you handle stress?",
                "answer": "I usually analyze the situation step by step"
            }
        ],
        "languages": "en"
    }
    
    result4 = analyzer.analyze(**case4)
    response4 = json.loads(result4["content"])
    
    print(f"Status: {response4.get('status')}")
    print(f"Expected: incomplete/complete (✅ CORRECT)" if response4.get('status') in ['incomplete', 'complete'] else f"Expected: incomplete/complete (❌ WRONG)")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✅ Test 1 (Identity in past): {'PASSED' if response1.get('status') == 'incomplete' else 'FAILED'}")
    print(f"✅ Test 2 (Identity as last): {'PASSED' if response2.get('status') == 'identity' else 'FAILED'}")
    print(f"✅ Test 3 (Standalone identity): {'PASSED' if response3.get('status') == 'identity' else 'FAILED'}")
    print(f"✅ Test 4 (Ignore user_input when new_input exists): {'PASSED' if response4.get('status') in ['incomplete', 'complete'] else 'FAILED'}")
    
    all_passed = (
        response1.get('status') == 'incomplete' and
        response2.get('status') == 'identity' and
        response3.get('status') == 'identity' and
        response4.get('status') in ['incomplete', 'complete']
    )
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Logic is correct!")
    else:
        print("\n❌ Some tests failed. Check the logic.")

if __name__ == "__main__":
    test_corrected_logic()
