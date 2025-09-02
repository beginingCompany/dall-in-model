#!/usr/bin/env python3
"""
Test the restructured logic with user examples.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app.personality_analyzer import PersonalityAnalyzer

def test_restructured_logic():
    """Test the restructured logic as described by the user."""
    analyzer = PersonalityAnalyzer()
    
    print("=" * 70)
    print("RESTRUCTURED LOGIC TESTS")
    print("=" * 70)
    
    # Test 1: First example from user - greeting user_input + off-topic latest
    print("\n🧪 TEST 1: Greeting user_input + Off-topic latest answer")
    print("-" * 60)
    print("Scenario: user_input='مرحبا! انا احمد كيف حالك؟' (greeting)")
    print("          latest_answer='ما لون السماء' (off-topic)")
    print("Expected: Should prioritize LATEST answer (off-topic) and provide varied casual response")
    
    result1 = analyzer.analyze(
        id=102,
        user_input="مرحبا! انا احمد كيف حالك؟",
        new_input=[
            {"question": "هل يمكنك أن تخبرني المزيد عن نفسك؟", "answer": "من انت وما هدفك؟"},
            {"question": "هل يمكنك أن تخبرني المزيد عن نفسك؟", "answer": "ما لون السماء"}
        ],
        languages="ar"
    )
    
    print(f"Result - Off-topic: '{result1['personal_greeting_and_off_topic']}'")
    print(f"Result - Identity: '{result1['description_identity']}'")
    print(f"Status: {result1['status']}")
    
    if result1['personal_greeting_and_off_topic'] and not result1['description_identity']:
        print("✅ SUCCESS: Off-topic latest answer prioritized correctly")
    else:
        print("❌ FAIL: Off-topic latest answer not prioritized")
    
    # Test 2: Second example from user - identity user_input + off-topic latest  
    print("\n🧪 TEST 2: Identity user_input + Off-topic latest answer")
    print("-" * 60)
    print("Scenario: user_input='من انت وما هدفك؟' (identity)")
    print("          latest_answer='ما لون السماء' (off-topic)")
    print("Expected: Should prioritize LATEST answer (off-topic) and provide varied casual response")
    
    result2 = analyzer.analyze(
        id=102,
        user_input="من انت وما هدفك؟",
        new_input=[
            {"question": "هل يمكنك أن تخبرني المزيد عن نفسك؟", "answer": "من انت وما هدفك؟"},
            {"question": "هل يمكنك أن تخبرني المزيد عن نفسك؟", "answer": "ما لون السماء"}
        ],
        languages="ar"
    )
    
    print(f"Result - Off-topic: '{result2['personal_greeting_and_off_topic']}'")
    print(f"Result - Identity: '{result2['description_identity']}'")
    print(f"Status: {result2['status']}")
    
    if result2['personal_greeting_and_off_topic'] and not result2['description_identity']:
        print("✅ SUCCESS: Off-topic latest answer prioritized correctly over identity user_input")
    else:
        print("❌ FAIL: Off-topic latest answer not prioritized")
        
    # Test 3: Personality accumulation example
    print("\n🧪 TEST 3: Personality accumulation scenario")
    print("-" * 60)
    print("Scenario: user_input='I am ahmed, I am critical thinker, I like go out with friends'")
    print("          latest_answer='I like go out with friends' (personality)")
    print("Expected: Should continue with personality analysis")
    
    result3 = analyzer.analyze(
        id=103,
        user_input="I am ahmed, I am critical thinker, I like go out with friends",
        new_input=[
            {"question": "Tell me about yourself", "answer": "I am ahmed"},
            {"question": "More details", "answer": "I am critical thinker"},
            {"question": "Anything else", "answer": "I like go out with friends"}
        ],
        languages="en"
    )
    
    print(f"Result - Off-topic: '{result3['personal_greeting_and_off_topic']}'")
    print(f"Result - Identity: '{result3['description_identity']}'")
    print(f"Status: {result3['status']}")
    print(f"Missing traits: {result3['missing_traits']}")
    
    if not result3['personal_greeting_and_off_topic'] and not result3['description_identity'] and result3['status'] == 'incomplete':
        print("✅ SUCCESS: Personality analysis continued correctly")
    else:
        print("❌ FAIL: Personality analysis not handled correctly")
    
    print("\n" + "=" * 70)
    print("RESTRUCTURED LOGIC TESTS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_restructured_logic()
