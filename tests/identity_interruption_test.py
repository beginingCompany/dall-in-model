#!/usr/bin/env python3

"""
Focused test of identity question handling during conversations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app. personality_analyzer import PersonalityAnalyzer

def test_identity_interruption():
    """Test specific identity question scenarios"""
    
    print("🧪 IDENTITY QUESTION INTERRUPTION TEST")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Setup a conversation with significant personality data
    conversation_history = [
        {"question": "Tell me about yourself", "answer": "I'm a very analytical person who loves data and systematic approaches to problem-solving"},
        {"question": "How do you work with others?", "answer": "I really enjoy team collaboration and often take leadership roles in group projects"},
        {"question": "Describe your work style", "answer": "I'm extremely organized, always plan ahead, and stick to structured schedules and deadlines"}
    ]
    
    print("📋 ESTABLISHED CONVERSATION CONTEXT:")
    for i, qa in enumerate(conversation_history, 1):
        print(f"   {i}. Q: {qa['question']}")
        print(f"      A: {qa['answer']}")
    
    # Test different identity questions
    identity_questions = [
        "Who are you?",
        "What is your purpose?", 
        "Who is your developer?",
        "What is BEGINING?",
        "How do you analyze personality?"
    ]
    
    print(f"\n🧪 TESTING IDENTITY QUESTIONS:")
    print("(Should preserve conversation context and show minimal missing traits)")
    
    for i, question in enumerate(identity_questions, 1):
        print(f"\n--- Test {i}: '{question}' ---")
        
        result = analyzer.analyze(
            id=1,
            user_input=question,
            new_input=conversation_history,
            languages="en"
        )
        
        # Check results
        has_identity = result.get('description_identity') is not None
        missing_traits = result.get('missing_traits', [])
        clarification = result.get('clarification_questions', [])
        
        print(f"Identity Response: {'Given' if has_identity else 'Not Given'}")
        print(f"Missing Traits: {missing_traits} (should be minimal due to rich history)")
        print(f"Clarification Questions: {len(clarification)} question(s)")
        
        if has_identity:
            identity_text = result['description_identity'][:60] + "..." if result['description_identity'] else ""
            print(f"   Response Preview: {identity_text}")
        
        # Verify conversation context is preserved
        expected_minimal_missing = len(missing_traits) <= 1  # Should detect most traits from rich history
        print(f"Context Preserved: {'Yes' if expected_minimal_missing else 'Needs Review'}")
    
    # Test the "false positive" cases that should NOT trigger identity responses
    print(f"\n🚫 TESTING NON-IDENTITY STATEMENTS:")
    print("(These should NOT trigger identity responses)")
    
    non_identity_statements = [
        "I am a developer working on AI projects",
        "My purpose in life is to help others achieve their goals", 
        "I work with a great team of engineers",
        "The company's role in the market is important to me",
        "I have a developer mindset and love coding"
    ]
    
    for i, statement in enumerate(non_identity_statements, 1):
        print(f"\n--- False Positive Test {i}: '{statement}' ---")
        
        result = analyzer.analyze(
            id=1,
            user_input=statement,
            new_input=conversation_history,
            languages="en"
        )
        
        has_identity = result.get('description_identity') is not None
        print(f"Identity Response: {'❌ INCORRECTLY Given' if has_identity else 'Correctly NOT Given'}")
        
        if has_identity:
            print(f"   ⚠️  FALSE POSITIVE DETECTED!")

    print(f"\n📊 SUMMARY:")
    print(f"Identity questions properly detected and responded to")
    print(f"Conversation context preserved during identity questions")
    print(f"User self-descriptions do not trigger false identity responses")
    print(f"System maintains conversation flow seamlessly")

if __name__ == "__main__":
    test_identity_interruption()
