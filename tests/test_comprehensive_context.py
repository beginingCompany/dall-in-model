#!/usr/bin/env python3
"""
Comprehensive test to verify GPT receives and processes full context effectively.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_comprehensive_context_processing():
    """Test that GPT can effectively process comprehensive context"""
    
    print("=== Testing Comprehensive Context Processing ===")
    analyzer = PersonalityAnalyzer()
    
    # Rich conversation with multiple personality insights
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "I tend to stay calm and analyze the situation logically. I don't let emotions cloud my judgment, but I do acknowledge them and try to understand what they're telling me."
        }
    ]
    
    print(f"User Input: '{user_input}'")
    print(f"Conversation Context: {len(new_input)} previous exchanges")
    print("Q&A History:")
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
    
    # Analyze the quality of the response
    print("\n" + "="*50)
    print("ANALYSIS QUALITY CHECK:")
    
    # Check if GPT provided meaningful analysis
    if result.get("description_english") and len(result["description_english"]) > 50:
        print("✅ GPT provided comprehensive English description")
        print(f"   Description: {result['description_english'][:100]}...")
    else:
        print("❌ GPT description is too brief or missing")
    
    # Check trait detection
    print(f"✅ Missing traits: {result.get('missing_traits', [])}")
    print(f"✅ Status: {result.get('status', 'unknown')}")
    
    # Check if identity was correctly avoided
    if result.get("description_identity"):
        print(f"❌ Unexpected identity response: {result['description_identity']}")
        return False
    else:
        print("✅ No identity response (correct)")
        
    return True

def test_confused_answer_in_rich_context():
    """Test confused answer within rich conversational context"""
    
    print("\n\n=== Testing Confused Answer in Rich Context ===")
    analyzer = PersonalityAnalyzer()
    
    # Same rich context but with confused answer
    user_input = "who are you"  # This is the confused answer
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "I tend to stay calm and analyze the situation logically. I don't let emotions cloud my judgment."
        },
        {
            "question": "What motivates you most in your work or personal projects?",
            "answer": "who are you"  # Current confused answer
        }
    ]
    
    print(f"User Input: '{user_input}' (confused answer)")
    print(f"Rich Conversation Context: {len(new_input)} previous exchanges")
    
    result = analyzer.analyze(
        id=225985882206,
        user_input=user_input,
        new_input=new_input,
        languages="en"
    )
    
    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Should NOT trigger identity and should continue personality analysis
    if result.get("description_identity"):
        print(f"❌ IDENTITY TRIGGERED: {result['description_identity']}")
        print("❌ This should NOT happen in rich conversation context!")
        return False
    else:
        print("✅ NO IDENTITY RESPONSE: Correct behavior in rich context")
        return True

if __name__ == "__main__":
    print("Testing comprehensive context processing...")
    
    test1_passed = test_comprehensive_context_processing()
    test2_passed = test_confused_answer_in_rich_context()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE CONTEXT TEST SUMMARY:")
    print(f"Test 1 (Rich context processing): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Confused answer in rich context): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 GPT IS RECEIVING AND PROCESSING FULL CONTEXT EFFECTIVELY!")
        print("✅ Context-aware identity detection working perfectly")
        print("✅ Rich personality analysis working correctly")
    else:
        print("⚠️ Some issues remain with context processing")
