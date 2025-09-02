"""
Test the actual scenario from the user's conversation example.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def test_real_scenario():
    """Test the exact scenario the user described."""
    
    print("🎯 Testing Real User Scenario")
    print("=" * 40)
    
    analyzer = PersonalityAnalyzer()
    
    # Simulate a real conversation where someone asks "who are you" 
    # as an answer to a clarification question
    print("Scenario: User is in middle of personality analysis")
    print("They get asked: 'How do you handle stressful situations?'")
    print("They respond with: 'who are you' (confused/deflecting)")
    
    # This simulates the conversation state when user responds with "who are you"
    # as an answer to a clarification question
    result = analyzer.analyze(
        id=123,
        user_input="who are you",  # User's confused response
        new_input=[
            {"question": "Tell me about your interests", "answer": "I like technology and programming"},
            {"question": "How do you handle stressful situations?", "answer": ""}  # Current question, no answer yet
        ],
        languages="en"
    )
    
    print(f"\nResult:")
    print(f"Has identity response: {'description_identity' in result and result['description_identity']}")
    print(f"Status: {result.get('status', 'unknown')}")
    if result.get('description_identity'):
        print(f"Identity response: {result['description_identity'][:100]}...")
        print("❌ FALSE POSITIVE: Identity detected in conversation context")
    else:
        print("✅ CORRECT: No identity response triggered")
    
    # Test with even more conversation history
    print(f"\n" + "="*40)
    print("Scenario 2: Longer conversation history")
    
    result2 = analyzer.analyze(
        id=456,
        user_input="who are you",
        new_input=[
            {"question": "What are your hobbies?", "answer": "I enjoy reading and sports"},
            {"question": "How do you work with others?", "answer": "I'm collaborative and helpful"},
            {"question": "What motivates you?", "answer": "Learning new things"},
            {"question": "How do you handle pressure?", "answer": ""}  # Current question
        ],
        languages="en"
    )
    
    print(f"Has identity response: {'description_identity' in result2 and result2['description_identity']}")
    if result2.get('description_identity'):
        print("❌ FALSE POSITIVE: Still triggering with more context")
    else:
        print("✅ CORRECT: No identity response with longer conversation")

if __name__ == "__main__":
    test_real_scenario()
