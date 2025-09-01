"""
Debug the exact scenario from the user's example
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer
import json

def debug_user_scenario():
    """Debug the exact scenario the user provided."""
    
    print("🐛 DEBUG: User's Exact Scenario")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Exact input from user
    input_data = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "who are you"
            },
            {
                "question": "What brings you the most joy or satisfaction in your life, and how do you express those feelings?",
                "answer": "play gaminig "
            },
            {
                "question": "When making important decisions, do you rely more on facts and logic or intuition and personal values?",
                "answer": " i always rely on facts and logic "
            }
        ],
        "languages": "en"
    }
    
    print("Input Analysis:")
    print(f"- User input: '{input_data['user_input'][:60]}...'")
    print(f"- Conversation history: {len(input_data['new_input'])} exchanges")
    print(f"- Answer that triggered issue: '{input_data['new_input'][1]['answer']}'")
    print()
    
    # Test the get_identity_response method directly first
    conversation_context = []
    for qa in input_data['new_input']:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if q:
            if a:
                conversation_context.append(f"Q: {q}\nA: {a}")
            else:
                conversation_context.append(f"Q: {q}\nA: [pending]")
    
    print("Built conversation context:")
    for i, ctx in enumerate(conversation_context):
        print(f"{i+1}. {ctx}")
    print()
    
    # Test direct method
    direct_result = PersonalityAnalyzer.get_identity_response(
        input_data['user_input'],
        language="en",
        openai_client=analyzer.client,
        conversation_context=conversation_context
    )
    
    print(f"Direct get_identity_response: {'Found' if direct_result else 'Not found'}")
    print()
    
    # Test full analyze method
    result = analyzer.analyze(
        id=input_data['id'],
        user_input=input_data['user_input'],
        new_input=input_data['new_input'],
        languages=input_data['languages']
    )
    
    print("Full analyze result:")
    print(f"- Has identity response: {bool(result.get('description_identity'))}")
    print(f"- Status: {result.get('status')}")
    if result.get('description_identity'):
        print(f"- Identity response: {result['description_identity'][:80]}...")
        print("❌ PROBLEM: Identity response triggered inappropriately!")
    else:
        print("✅ CORRECT: No identity response")

if __name__ == "__main__":
    debug_user_scenario()
