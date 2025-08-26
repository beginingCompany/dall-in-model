#!/usr/bin/env python3
"""
Test handle_answer function in both standalone and conversation contexts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_handle_answer_comprehensive():
    """Test handle_answer in different contexts"""
    
    print("=== Testing handle_answer - Comprehensive ===\n")
    
    # Test 1: Standalone identity questions (should trigger identity responses)
    print("1. STANDALONE IDENTITY QUESTIONS:")
    print("-" * 40)
    
    standalone_tests = ["who are you", "what is begining", "what is your purpose"]
    
    for trigger in standalone_tests:
        print(f"Testing standalone: '{trigger}'")
        
        result = PersonalityAnalyzer.handle_answer(
            question="Initial question",
            answer=trigger,
            context=[]  # No conversation context
        )
        
        print(f"  Type: {result.get('type')}")
        print(f"  Skip traits: {result.get('skip_traits')}")
        if result.get('description_identity'):
            print(f"  Identity: {result['description_identity'][:50]}...")
        print()
    
    # Test 2: Identity questions in conversation (should be filtered/treated as personality)
    print("2. IDENTITY QUESTIONS IN CONVERSATION:")
    print("-" * 40)
    
    conversation_context = [
        "Q: How do you interact with others?\nA: I love teamwork and leadership roles"
    ]
    
    for trigger in standalone_tests:
        print(f"Testing in conversation: '{trigger}'")
        
        result = PersonalityAnalyzer.handle_answer(
            question="How do you handle emotions?",
            answer=trigger,
            context=conversation_context
        )
        
        print(f"  Type: {result.get('type')}")
        print(f"  Skip traits: {result.get('skip_traits')}")
        if result.get('detected_traits'):
            print(f"  Detected traits: {result['detected_traits']}")
        print()
    
    # Test 3: Normal personality answers
    print("3. NORMAL PERSONALITY ANSWERS:")
    print("-" * 40)
    
    personality_answers = [
        "I stay calm and think logically",
        "I love working with teams",
        "I'm very organized and methodical"
    ]
    
    for answer in personality_answers:
        print(f"Testing personality: '{answer}'")
        
        result = PersonalityAnalyzer.handle_answer(
            question="How do you handle challenges?",
            answer=answer,
            context=conversation_context
        )
        
        print(f"  Type: {result.get('type')}")
        print(f"  Detected traits: {result.get('detected_traits')}")
        print()

if __name__ == "__main__":
    test_handle_answer_comprehensive()
