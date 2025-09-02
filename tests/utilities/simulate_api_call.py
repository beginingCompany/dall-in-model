"""
Simulate the exact API call to verify our implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer
import json

def simulate_api_call():
    """Simulate the exact API call format."""
    
    print("🌐 SIMULATING EXACT API CALL")
    print("=" * 50)
    
    # Create analyzer instance (same as API)
    analyzer = PersonalityAnalyzer()
    
    # Exact request data from user
    request_data = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        "new_input": [
            {
                "question": "How do you usually interact with others in group settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "who are you"
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "i analytical can solving the problems by analyze them "
            }
        ],
        "languages": "en"
    }
    
    print("REQUEST:")
    print(json.dumps(request_data, indent=2))
    print()
    
    # Call analyzer exactly like API does
    result = analyzer.analyze(
        id=request_data["id"],
        user_input=request_data["user_input"],
        new_input=request_data["new_input"],
        languages=request_data["languages"]
    )
    
    print("RESPONSE:")
    print(json.dumps(result, indent=2))
    print()
    
    # Analysis
    has_identity = bool(result.get('description_identity'))
    print("ANALYSIS:")
    print(f"✅ Has identity response: {has_identity}")
    
    if has_identity:
        print("❌ UNEXPECTED: Identity response found")
        print(f"   Response: {result['description_identity']}")
    else:
        print("✅ EXPECTED: No identity response (correctly filtered)")
    
    print(f"✅ Status: {result.get('status')}")
    print(f"✅ Missing traits: {result.get('missing_traits', [])}")
    print(f"✅ Clarification questions: {len(result.get('clarification_questions', []))} questions")
    
    # Compare with user's expected result
    user_expected = {
        "id": 225985882206,
        "status": "incomplete",
        "description_arabic": "",
        "description_english": "",
        "description_identity": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential. Let's get started by discovering a bit about you.",
        "missing_traits": ["emotional"],
        "clarification_questions": ["What brings you the most joy or satisfaction in your life, and how do you express those feelings?"],
        "input_tokens": 58,
        "output_tokens": 35,
        "total_tokens": 93
    }
    
    print(f"\n📊 COMPARISON WITH USER'S RESULT:")
    print(f"User expected identity response: {bool(user_expected.get('description_identity'))}")
    print(f"Our result identity response: {has_identity}")
    print(f"Match: {'❌ DIFFERENT' if bool(user_expected.get('description_identity')) != has_identity else '✅ SAME'}")

if __name__ == "__main__":
    simulate_api_call()
