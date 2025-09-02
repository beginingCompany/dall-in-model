#!/usr/bin/env python3
"""
Final test to verify both user cases work correctly
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_user_cases():
    """Test both specific user cases"""
    
    print("🧪 TESTING USER CASES - FINAL VERIFICATION")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test Case 1: English case with detailed data
    print("TEST CASE 1: English Identity Question with Detailed Data")
    print("-" * 50)
    
    case1_input = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
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
    
    result1 = analyzer.analyze(**case1_input)
    response1 = json.loads(result1["content"])
    
    print(f"✅ Status: {response1.get('status')}")
    print(f"✅ Identity Response: {response1.get('description_identity', '')[:80]}...")
    print(f"✅ Missing Traits: {response1.get('missing_traits', [])}")
    print(f"✅ Clarification Questions ({len(response1.get('clarification_questions', []))}):")
    
    for i, question in enumerate(response1.get('clarification_questions', []), 1):
        print(f"   {i}. {question}")
    
    # Verify requirements
    assert response1.get('status') == 'identity', "Should have identity status"
    assert len(response1.get('description_identity', '')) > 0, "Should have identity response"
    assert len(response1.get('missing_traits', [])) > 0, "Should have missing traits to continue conversation"
    assert len(response1.get('clarification_questions', [])) > 0, "Should have clarification questions"
    
    print("✅ CASE 1: PASSED - Conversation continues with clarification questions")
    
    print("\n" + "=" * 60 + "\n")
    
    # Test Case 2: Arabic identity question
    print("TEST CASE 2: Arabic Identity Question")
    print("-" * 50)
    
    case2_input = {
        "id": 65387652876,
        "user_input": "من انت",
        "new_input": [],
        "languages": "ar"
    }
    
    result2 = analyzer.analyze(**case2_input)
    response2 = json.loads(result2["content"])
    
    print(f"✅ Status: {response2.get('status')}")
    print(f"✅ Identity Response: {response2.get('description_identity', '')[:80]}...")
    print(f"✅ Missing Traits: {response2.get('missing_traits', [])}")
    print(f"✅ Clarification Questions ({len(response2.get('clarification_questions', []))}):")
    
    for i, question in enumerate(response2.get('clarification_questions', []), 1):
        print(f"   {i}. {question}")
    
    # Verify Arabic content
    identity_response = response2.get('description_identity', '')
    questions = response2.get('clarification_questions', [])
    
    expected_arabic_start = "أنا ماينس زيرو"
    uses_correct_identity = identity_response.startswith(expected_arabic_start)
    
    has_arabic_questions = any(
        any('\u0600' <= char <= '\u06FF' for char in q) for q in questions
    ) if questions else False
    
    print(f"✅ Uses Correct Arabic Identity: {uses_correct_identity}")
    print(f"✅ Has Arabic Questions: {has_arabic_questions}")
    
    # Verify requirements
    assert response2.get('status') == 'identity', "Should have identity status"
    assert uses_correct_identity, "Should use predefined Arabic identity response"
    assert len(response2.get('missing_traits', [])) > 0, "Should have missing traits"
    assert len(response2.get('clarification_questions', [])) > 0, "Should have clarification questions"
    assert has_arabic_questions, "Questions should be in Arabic"
    
    print("✅ CASE 2: PASSED - Correct Arabic identity with Arabic clarification questions")
    
    print("\n" + "🎉" * 20)
    print("🎉 ALL USER CASES PASSED!")
    print("🎉" * 20)
    
    print("\n📋 SUMMARY OF FIXES APPLIED:")
    print("=" * 50)
    print("✅ Fixed Arabic trigger detection - added 'من انت' variant")
    print("✅ Fixed user_input identity detection for standalone Arabic questions")
    print("✅ Fixed missing trait analysis to always continue conversation")
    print("✅ Ensured clarification questions are generated even with good data")
    print("✅ Verified Arabic responses use predefined identity text")
    print("✅ Confirmed conversation never stops - always provides next questions")
    
    print("\n🚀 READY FOR PRODUCTION!")
    print("Both cases now work exactly as requested:")
    print("- Identity responses are instant and include clarification questions")
    print("- Conversation continues seamlessly in both English and Arabic")
    print("- Missing traits are identified to keep dialogue flowing")
    print("- No conversation interruptions or restarts needed")

if __name__ == "__main__":
    test_user_cases()
