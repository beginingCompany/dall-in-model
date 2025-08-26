"""
Test with "who are you" as the main user_input in conversation context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def test_who_are_you_as_main_input():
    """Test 'who are you' as main user_input with conversation context."""
    
    print("🧪 TEST: 'who are you' as main user_input")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Test scenario: user_input is "who are you" but there's conversation history
    result = analyzer.analyze(
        id=123,
        user_input="who are you",  # This is the problematic input
        new_input=[
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and taking leadership roles"
            },
            {
                "question": "How do you typically approach complex problems?", 
                "answer": ""  # Current question being answered
            }
        ],
        languages="en"
    )
    
    print(f"Result with 'who are you' as main input:")
    print(f"- Has identity response: {bool(result.get('description_identity'))}")
    print(f"- Status: {result.get('status')}")
    
    if result.get('description_identity'):
        print(f"- Identity response: {result['description_identity'][:80]}...")
        print("❌ PROBLEM: Still triggering identity response!")
    else:
        print("✅ CORRECT: No identity response triggered")
    
    # Also test the direct method
    conversation_context = [
        "Q: How do you usually interact with others in social settings?\nA: I love working in teams and taking leadership roles",
        "Q: How do you typically approach complex problems?\nA: [pending]"
    ]
    
    direct_result = PersonalityAnalyzer.get_identity_response(
        "who are you",
        language="en", 
        openai_client=analyzer.client,
        conversation_context=conversation_context
    )
    
    print(f"\nDirect method result: {'Found' if direct_result else 'Not found'}")

if __name__ == "__main__":
    test_who_are_you_as_main_input()
