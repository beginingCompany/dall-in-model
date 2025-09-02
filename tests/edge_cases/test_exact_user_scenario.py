#!/usr/bin/env python3
"""
<<<<<<< HEAD
Test the exact scenario reported by the user to verify the fix is working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_exact_user_scenario():
    """Test the exact scenario from the user's request"""
    
    print("=== Testing Exact User Scenario ===")
    analyzer = PersonalityAnalyzer()
    
    # This is the exact data the user provided
    user_input = "who are you"  # This is the confused answer to emotional question
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "who are you"  # <- This is the current input, a confused answer
        }
    ]
    
    # The user also had this analytical answer to the same question:
    # "i analytical can solving the problems by analyze them"
    # But let's test the "who are you" case first
    
    print(f"User Input: '{user_input}'")
    print(f"Conversation Context: {len(new_input)} previous exchanges")
    print("Previous Q&A:")
    for i, qa in enumerate(new_input, 1):
        print(f"  {i}. Q: {qa['question']}")
        print(f"     A: {qa['answer']}")
    
    print("\n" + "="*50)
    print("ANALYZING...")
    
    result = analyzer.analyze(
        id=225985882206,
        user_input=user_input,
        new_input=new_input,
        languages="en"
    )
    
    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Check if identity response was triggered
    if result.get("description_identity"):
        print(f"\n❌ IDENTITY DETECTED: {result['description_identity']}")
        print("❌ This should NOT happen in conversation context!")
        return False
    else:
        print(f"\n✅ NO IDENTITY RESPONSE: Good!")
        print("✅ System correctly treated 'who are you' as confused answer, not identity question")
        return True

def test_with_full_context():
    """Test with the complete context including the analytical answer"""
    
    print("\n\n=== Testing With Full Context ===")
    analyzer = PersonalityAnalyzer()
    
    # Complete scenario with both answers
    user_input = "i analytical can solving the problems by analyze them"  
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "who are you"  # Previous confused answer
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?", 
            "answer": "i analytical can solving the problems by analyze them"  # Current analytical answer
        }
    ]
    
    print(f"User Input: '{user_input}'")
    print(f"Conversation Context: {len(new_input)} previous exchanges")
    
    result = analyzer.analyze(
        id=225985882206,
        user_input=user_input,
        new_input=new_input,
        languages="en"
    )
    
    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # This should not trigger identity and should process as personality data
    if result.get("description_identity"):
        print(f"\n❌ UNEXPECTED IDENTITY: {result['description_identity']}")
        return False
    else:
        print(f"\n✅ NO IDENTITY RESPONSE: Correct behavior")
        return True

if __name__ == "__main__":
    print("Testing the exact user scenario to verify context-aware identity detection...")
    
    test1_passed = test_exact_user_scenario()
    test2_passed = test_with_full_context()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Test 1 (Confused 'who are you' answer): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Analytical answer with context): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Context-aware identity detection is working correctly.")
    else:
        print("⚠️ Some tests failed. Identity detection needs further refinement.")
=======
Test the exact user scenario
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_exact_scenario():
    """Test the exact user scenario"""
    
    print("🧪 TESTING EXACT USER SCENARIO")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Exact user data
    test_data = {
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
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "i analytical can solving the problems by analyze them"
            }
        ],
        "languages": "en"
    }
    
    result = analyzer.analyze(**test_data)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Identity Response: {response.get('description_identity', '')}")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    print(f"Clarification Questions: {len(response.get('clarification_questions', []))}")
    
    for i, q in enumerate(response.get('clarification_questions', []), 1):
        print(f"  {i}. {q}")
    
    # Expected user output
    print("\n" + "=" * 50)
    print("USER EXPECTED OUTPUT:")
    expected = {
        "id": 225985882206,
        "status": "identity",
        "description_arabic": "",
        "description_english": "",
        "description_identity": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential. Let's get started by discovering a bit about you.",
        "missing_traits": [
            "emotional",
            "behavioral"
        ],
        "clarification_questions": [
            "What brings you the most joy or satisfaction in your life, and how do you express those feelings?",
            "Do you tend to plan activities in advance or prefer to be spontaneous with your time?"
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    print(json.dumps(expected, indent=2))
    
    # Verify match
    print("\n" + "=" * 50)
    print("VERIFICATION:")
    
    matches_status = response.get('status') == expected['status']
    has_identity = len(response.get('description_identity', '')) > 0
    has_missing = len(response.get('missing_traits', [])) > 0
    has_questions = len(response.get('clarification_questions', [])) > 0
    
    print(f"✅ Status matches: {matches_status}")
    print(f"✅ Has identity response: {has_identity}")
    print(f"✅ Has missing traits: {has_missing}")
    print(f"✅ Has clarification questions: {has_questions}")
    
    if all([matches_status, has_identity, has_missing, has_questions]):
        print("\n🎉 PERFECT! Scenario works exactly as user requested!")
        return True
    else:
        print("\n❌ Needs adjustment")
        return False

if __name__ == "__main__":
    success = test_exact_scenario()
    if success:
        print("\n🚀 SYSTEM READY - Identity responses with clarification questions working perfectly!")
    else:
        print("\n🔧 Needs fixes")
>>>>>>> f912c397f4608be37933b416c471652681384d61
