#!/usr/bin/env python3
"""
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
