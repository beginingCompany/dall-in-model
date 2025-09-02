#!/usr/bin/env python3
"""
Test the exact scenario that's still triggering identity responses incorrectly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_current_issue():
    """Test the exact scenario from the user's request"""
    
    print("=== Testing Current Issue ===")
    analyzer = PersonalityAnalyzer()
    
    # Exact data from user's request
    user_input = "who are you"  # This should NOT trigger identity response
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "who are you"  # This is a confused answer, not a genuine question
        }
    ]
    
    # Build conversation context as the analyze method does
    conversation_context = []
    for qa in new_input:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        # Only include Q&A pairs where the answer is NOT the current user_input being processed
        if q and a and a.strip() != user_input.strip():
            conversation_context.append(f"Q: {q}\nA: {a}")
    
    print(f"User Input: '{user_input}'")
    print(f"Conversation Context Length: {len(conversation_context)}")
    print("Conversation Context:")
    for i, ctx in enumerate(conversation_context, 1):
        print(f"  {i}. {ctx}")
    
    # Test the identity detection directly
    identity_response = analyzer.get_identity_response(
        user_input, 
        "en", 
        None,  # No OpenAI client for regex-only testing
        conversation_context=conversation_context
    )
    
    print(f"\nIdentity Response: '{identity_response}'")
    
    if identity_response:
        print("❌ PROBLEM: Identity response triggered when it shouldn't!")
        print("This 'who are you' is a confused answer in conversation context")
        return False
    else:
        print("✅ CORRECT: No identity response triggered")
        return True

def test_handle_answer_function():
    """Test the handle_answer function with the same scenario"""
    
    print("\n=== Testing handle_answer Function ===")
    
    question = "How do you typically approach and handle your emotions in challenging situations?"
    answer = "who are you"  # Confused answer
    context = [
        "Q: How do you usually interact with others in social settings?\nA: I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
    ]
    
    result = PersonalityAnalyzer.handle_answer(question, answer, context)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if result.get("type") == "identity":
        print("❌ handle_answer returned identity type when it should skip")
        return False
    else:
        print("✅ handle_answer correctly processed as traits")
        return True

if __name__ == "__main__":
    print("Testing the current identity detection issue...\n")
    
    test1_passed = test_current_issue()
    test2_passed = test_handle_answer_function()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Direct identity detection test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"handle_answer function test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if not test1_passed:
        print("\n🔧 DIAGNOSIS: Context-aware filtering needs adjustment")
        print("The issue is that 'who are you' is still matching identity triggers")
        print("even in conversation context where it should be ignored.")
    else:
        print("\n🎉 All tests passed! The system is working correctly.")
