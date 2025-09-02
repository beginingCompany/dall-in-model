"""
Test the exact scenario to confirm the issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def test_exact_issue():
    """Test the exact scenario from user's latest example."""
    
    analyzer = PersonalityAnalyzer()
    
    # User's EXACT scenario
    result = analyzer.analyze(
        id=225985882206,
        user_input="Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        new_input=[
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
        languages="en"
    )
    
    print("RESULT:")
    print(f"Has identity response: {bool(result.get('description_identity'))}")
    print(f"Status: {result.get('status')}")
    if result.get('description_identity'):
        print(f"Identity response: {result['description_identity']}")
        print("❌ ISSUE: Identity response triggered inappropriately")
    else:
        print("✅ CORRECT: No identity response")

if __name__ == "__main__":
    test_exact_issue()
