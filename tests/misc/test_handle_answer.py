#!/usr/bin/env python3
"""
Test the handle_answer function with various identity triggers in conversation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_handle_answer_with_identity_triggers():
    """Test handle_answer function with identity triggers"""
    
    print("=== Testing handle_answer with Identity Triggers ===\n")
    
    # Sample conversation context
    context = [
        "Q: How do you interact with others?\nA: I love teamwork and leadership roles"
    ]
    
    # Test various identity triggers
    test_cases = [
        "who are you",
        "what is begining", 
        "what is your purpose",
        "how do you analyze",
        "who is your developer"
    ]
    
    for trigger in test_cases:
        print(f"Testing: '{trigger}'")
        
        result = PersonalityAnalyzer.handle_answer(
            question="How do you handle emotions?",
            answer=trigger,
            context=context
        )
        
        print(f"Result type: {result.get('type')}")
        print(f"Skip traits: {result.get('skip_traits')}")
        if result.get('description_identity'):
            print(f"Identity response: {result['description_identity'][:50]}...")
        print(f"Reason: {result.get('reason', 'N/A')}")
        print("-" * 40)
    
    # Test with actual personality answer
    print("Testing with personality answer:")
    result = PersonalityAnalyzer.handle_answer(
        question="How do you handle emotions?",
        answer="I stay calm and think logically through problems",
        context=context
    )
    
    print(f"Result type: {result.get('type')}")
    print(f"Skip traits: {result.get('skip_traits')}")
    print(f"Detected traits: {result.get('detected_traits')}")
    print(f"Reason: {result.get('reason', 'N/A')}")

if __name__ == "__main__":
    test_handle_answer_with_identity_triggers()
