#!/usr/bin/env python3

"""
Test script for the improved personality analyzer
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_identity_detection():
    """Test the improved identity detection"""
    print("=== Testing Identity Detection ===")
    
    # Test cases that should NOT trigger identity responses
    non_identity_inputs = [
        "I am a developer who works with teams",
        "My purpose in life is to help others",
        "I like to work with others and help them",
        "I work as a developer in a team",
        "My role in the company is software engineer",
        "I have a purpose-driven mindset"
    ]
    
    # Test cases that SHOULD trigger identity responses
    identity_inputs = [
        "Who are you?",
        "What is your purpose?",
        "What do you do?",
        "Who is your developer?",
        "What is BEGINING?",
        "How do you analyze personality?"
    ]
    
    analyzer = PersonalityAnalyzer()
    
    print("\n--- Testing NON-identity inputs (should return empty) ---")
    for test_input in non_identity_inputs:
        response = analyzer.get_identity_response(test_input, "en", analyzer.client)
        result = "✓ PASS" if response == "" else f"✗ FAIL: Got '{response[:50]}...'"
        print(f"{result}: '{test_input}'")
    
    print("\n--- Testing identity inputs (should return responses) ---")
    for test_input in identity_inputs:
        response = analyzer.get_identity_response(test_input, "en", analyzer.client)
        result = "✓ PASS" if response != "" else "✗ FAIL: Got empty response"
        print(f"{result}: '{test_input}' -> '{response[:50]}...'")

def test_clarification_questions():
    """Test that only one clarification question is generated"""
    print("\n=== Testing Clarification Questions ===")
    
    analyzer = PersonalityAnalyzer()
    
    # Test with multiple missing traits
    missing_traits = ["emotional", "social", "cognitive", "behavioral"]
    questions = analyzer.generate_clarification_questions(missing_traits, "english")
    
    print(f"Missing traits: {missing_traits}")
    print(f"Generated questions: {len(questions)} (should be 1)")
    print(f"Questions: {questions}")
    
    if len(questions) == 1:
        print("✓ PASS: Only one question generated")
    else:
        print(f"✗ FAIL: Generated {len(questions)} questions instead of 1")

def test_conversation_continuity():
    """Test that conversation history is preserved during identity questions"""
    print("\n=== Testing Conversation Continuity ===")
    
    analyzer = PersonalityAnalyzer()
    
    # Simulate a conversation with personality info
    conversation_history = [
        {"question": "Tell me about yourself", "answer": "I am very analytical and logical in my approach to problems"},
        {"question": "How do you interact with others?", "answer": "I prefer working in teams and helping colleagues"}
    ]
    
    # User asks identity question mid-conversation
    identity_question = "What is your purpose?"
    
    result = analyzer.analyze(
        id=1,
        user_input=identity_question,
        new_input=conversation_history,
        languages="en"
    )
    
    print(f"Input: '{identity_question}'")
    print(f"Identity response given: {result.get('description_identity') is not None}")
    print(f"Missing traits: {result.get('missing_traits', [])}")
    print(f"Clarification questions: {len(result.get('clarification_questions', []))}")
    
    # Should detect cognitive and social traits from history
    expected_missing = ["emotional", "behavioral"]  # Only these should be missing
    actual_missing = result.get('missing_traits', [])
    
    if set(actual_missing) == set(expected_missing):
        print("✓ PASS: Conversation history properly preserved")
    else:
        print(f"✗ FAIL: Expected {expected_missing}, got {actual_missing}")

if __name__ == "__main__":
    test_identity_detection()
    test_clarification_questions()
    test_conversation_continuity()
    print("\n=== Test Summary ===")
    print("All tests completed!")
