#!/usr/bin/env python3
"""
Test both genuine identity questions and confused answers to verify the system works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_genuine_vs_confused():
    """Test genuine identity question vs confused answer in conversation"""
    
    print("=== Testing Genuine vs Confused Identity Questions ===")
    analyzer = PersonalityAnalyzer()
    
    # Test 1: Genuine identity question (standalone)
    print("\n1. GENUINE IDENTITY QUESTION (standalone):")
    result1 = analyzer.analyze(
        id=1,
        user_input="who are you",
        new_input=[],  # No conversation context
        languages="en"
    )
    print(f"Result: description_identity = {result1.get('description_identity', 'null')}")
    
    # Test 2: Confused answer in conversation
    print("\n2. CONFUSED ANSWER IN CONVERSATION:")
    result2 = analyzer.analyze(
        id=2,
        user_input="who are you",
        new_input=[
            {
                "question": "How do you handle stress?",
                "answer": "I stay calm and think logically"
            },
            {
                "question": "How do you work with others?",
                "answer": "who are you"  # Confused answer
            }
        ],
        languages="en"
    )
    print(f"Result: description_identity = {result2.get('description_identity', 'null')}")
    
    # Test 3: Different genuine identity question
    print("\n3. GENUINE IDENTITY QUESTION (what is your purpose):")
    result3 = analyzer.analyze(
        id=3,
        user_input="what is your purpose",
        new_input=[],
        languages="en"
    )
    print(f"Result: description_identity = {result3.get('description_identity', 'null')}")
    
    # Analysis
    print("\n" + "="*60)
    print("EXPECTED BEHAVIOR:")
    print("1. Standalone 'who are you' → SHOULD trigger identity response")
    print("2. 'who are you' in conversation → should NOT trigger identity response")
    print("3. 'what is your purpose' standalone → SHOULD trigger identity response")
    
    print("\nACTUAL BEHAVIOR:")
    print(f"1. Standalone: {'✅ Identity triggered' if result1.get('description_identity') else '❌ No identity'}")
    print(f"2. Conversation: {'❌ Identity triggered' if result2.get('description_identity') else '✅ No identity'}")
    print(f"3. Purpose question: {'✅ Identity triggered' if result3.get('description_identity') else '❌ No identity'}")
    
    # Check if behavior is correct
    correct_behavior = (
        bool(result1.get('description_identity')) and  # Should trigger for standalone
        not bool(result2.get('description_identity')) and  # Should NOT trigger in conversation
        bool(result3.get('description_identity'))  # Should trigger for purpose question
    )
    
    return correct_behavior

if __name__ == "__main__":
    print("Testing genuine vs confused identity questions...\n")
    
    success = test_genuine_vs_confused()
    
    print("\n" + "="*60)
    if success:
        print("🎉 PERFECT! The system correctly distinguishes between:")
        print("✅ Genuine identity questions (triggers response)")
        print("✅ Confused answers in conversation (skips response)")
        print("\n🚀 The context-aware filtering is working flawlessly!")
    else:
        print("❌ Some issues detected with identity question handling")
        print("🔧 The system needs further adjustment")
