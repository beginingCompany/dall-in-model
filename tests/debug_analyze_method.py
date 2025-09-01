"""
Debug the conversation context in the analyze method
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

# Monkey patch to add debug output
original_get_identity_response = PersonalityAnalyzer.get_identity_response

@staticmethod
def debug_get_identity_response(user_input: str, language: str = "en", openai_client=None, conversation_context: list = None) -> str:
    print(f"🔍 DEBUG: get_identity_response called")
    print(f"   Input: '{user_input}'")
    print(f"   Context length: {len(conversation_context) if conversation_context else 0}")
    if conversation_context:
        print(f"   Context: {conversation_context}")
    
    # Call original method
    result = original_get_identity_response(user_input, language, openai_client, conversation_context)
    
    print(f"   Result: {'Found' if result else 'Not found'}")
    if result:
        print(f"   Response: {result[:50]}...")
    print()
    
    return result

# Replace method
PersonalityAnalyzer.get_identity_response = debug_get_identity_response

def test_with_debug():
    analyzer = PersonalityAnalyzer()
    
    print("🧪 Debug Test: Real Scenario Analysis")
    print("=" * 50)
    
    result = analyzer.analyze(
        id=123,
        user_input="who are you",
        new_input=[
            {"question": "Tell me about your interests", "answer": "I like technology"},
            {"question": "How do you handle stress?", "answer": ""}
        ],
        languages="en"
    )
    
    print(f"Final result has identity: {'description_identity' in result and result['description_identity']}")

if __name__ == "__main__":
    test_with_debug()
