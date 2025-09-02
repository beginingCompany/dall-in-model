"""
Simulate the exact user scenario as an API call to verify the fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer
import json

def simulate_user_api_call():
    """Simulate the exact API call that was problematic for the user."""
    
    print("🌐 SIMULATING USER'S EXACT API CALL")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # EXACT request that was showing identity response
    request_data = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "i analytical can solving the problems by analyze them"
            }
        ],
        "languages": "en"
    }
    
    print("REQUEST (same as user's problematic scenario):")
    print(json.dumps(request_data, indent=2))
    print()
    
    # Call analyzer exactly like API
    result = analyzer.analyze(
        id=request_data["id"],
        user_input=request_data["user_input"],
        new_input=request_data["new_input"],
        languages=request_data["languages"]
    )
    
    print("OUR RESULT:")
    print(json.dumps(result, indent=2))
    print()
    
    # Compare with user's problematic result
    user_problematic_result = {
        "id": 225985882206,
        "status": "complete",
        "description_arabic": "",
        "description_english": "",
        "description_identity": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential. Let's get started by discovering a bit about you.",
        "missing_traits": [],
        "clarification_questions": [],
        "input_tokens": 58,
        "output_tokens": 35,
        "total_tokens": 93
    }
    
    print("USER'S PROBLEMATIC RESULT (old behavior):")
    print(json.dumps(user_problematic_result, indent=2))
    print()
    
    # Analysis
    our_has_identity = bool(result.get('description_identity'))
    user_had_identity = bool(user_problematic_result.get('description_identity'))
    
    print("📊 COMPARISON:")
    print(f"User's old result had identity: {user_had_identity}")
    print(f"Our new result has identity: {our_has_identity}")
    
    if not our_has_identity and user_had_identity:
        print("✅ SUCCESS: We fixed the issue!")
        print("   - User's old code: Incorrectly triggered identity response")
        print("   - Our new code: Correctly filters out false identity triggers")
        print("   - Context-aware filtering is working perfectly!")
    elif our_has_identity and not user_had_identity:
        print("❌ REGRESSION: We broke something that was working")
    elif our_has_identity and user_had_identity:
        print("❌ ISSUE: Still triggering identity response")
    else:
        print("✅ BOTH CORRECT: No identity response (but this wasn't the user's issue)")
    
    print(f"\nOur result status: {result.get('status')}")
    print(f"Missing traits: {result.get('missing_traits', [])}")
    print(f"Clarification questions: {len(result.get('clarification_questions', []))}")

if __name__ == "__main__":
    simulate_user_api_call()
