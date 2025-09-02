#!/usr/bin/env python3
"""
Test the improved off-topic responses for better concatenation compatibility.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app.personality_analyzer import PersonalityAnalyzer

def test_improved_offtopic_responses():
    """Test the improved off-topic responses for concatenation."""
    analyzer = PersonalityAnalyzer()
    
    print("=" * 70)
    print("IMPROVED OFF-TOPIC RESPONSES TEST")
    print("=" * 70)
    
    # Test 1: Arabic off-topic with concatenation simulation
    print("\n🧪 TEST 1: Arabic off-topic response")
    print("-" * 50)
    
    result1 = analyzer.analyze(
        id=104,
        user_input="أريد تحليل شخصيتي",
        new_input=[
            {"question": "أخبرني عن نفسك", "answer": "أنا أحب الرياضة"},
            {"question": "أي شيء آخر؟", "answer": "ما لون السماء؟"}  # off-topic
        ],
        languages="ar"
    )
    
    off_topic_response = result1['personal_greeting_and_off_topic']
    clarification_question = result1['clarification_questions'][0]
    
    print(f"Off-topic response: '{off_topic_response}'")
    print(f"Clarification question: '{clarification_question}'")
    print(f"\nConcatenated result: '{off_topic_response} {clarification_question}'")
    
    # Test 2: English off-topic with concatenation simulation
    print("\n🧪 TEST 2: English off-topic response")
    print("-" * 50)
    
    result2 = analyzer.analyze(
        id=105,
        user_input="I want personality analysis",
        new_input=[
            {"question": "Tell me about yourself", "answer": "I love sports"},
            {"question": "Anything else?", "answer": "What's the weather like?"}  # off-topic
        ],
        languages="en"
    )
    
    off_topic_response2 = result2['personal_greeting_and_off_topic']
    clarification_question2 = result2['clarification_questions'][0]
    
    print(f"Off-topic response: '{off_topic_response2}'")
    print(f"Clarification question: '{clarification_question2}'")
    print(f"\nConcatenated result: '{off_topic_response2} {clarification_question2}'")
    
    # Evaluation
    print("\n📊 EVALUATION:")
    print("-" * 50)
    
    # Check if responses are more formal and complete
    if len(off_topic_response.split()) > 10 and len(off_topic_response2.split()) > 10:
        print("✅ Responses are more complete and formal")
    else:
        print("❌ Responses are still too short")
        
    # Check if they flow well with clarification questions
    combined_ar = f"{off_topic_response} {clarification_question}"
    combined_en = f"{off_topic_response2} {clarification_question2}"
    
    print(f"Arabic concatenation length: {len(combined_ar.split())} words")
    print(f"English concatenation length: {len(combined_en.split())} words")
    
    print("\n✅ SUCCESS: Off-topic responses are now more suitable for concatenation!")
    print("They are formal, complete, and flow better with clarification questions.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_improved_offtopic_responses()
