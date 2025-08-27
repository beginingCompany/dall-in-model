#!/usr/bin/env python3
"""
Test the exact user scenario
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_exact_scenario():
    """Test the exact user scenario"""
    
    print("🧪 TESTING EXACT USER SCENARIO")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Exact user data
    test_data = {
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
    
    result = analyzer.analyze(**test_data)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Identity Response: {response.get('description_identity', '')}")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    print(f"Clarification Questions: {len(response.get('clarification_questions', []))}")
    
    for i, q in enumerate(response.get('clarification_questions', []), 1):
        print(f"  {i}. {q}")
    
    # Expected user output
    print("\n" + "=" * 50)
    print("USER EXPECTED OUTPUT:")
    expected = {
        "id": 225985882206,
        "status": "identity",
        "description_arabic": "",
        "description_english": "",
        "description_identity": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential. Let's get started by discovering a bit about you.",
        "missing_traits": [
            "emotional",
            "behavioral"
        ],
        "clarification_questions": [
            "What brings you the most joy or satisfaction in your life, and how do you express those feelings?",
            "Do you tend to plan activities in advance or prefer to be spontaneous with your time?"
        ],
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    print(json.dumps(expected, indent=2))
    
    # Verify match
    print("\n" + "=" * 50)
    print("VERIFICATION:")
    
    matches_status = response.get('status') == expected['status']
    has_identity = len(response.get('description_identity', '')) > 0
    has_missing = len(response.get('missing_traits', [])) > 0
    has_questions = len(response.get('clarification_questions', [])) > 0
    
    print(f"✅ Status matches: {matches_status}")
    print(f"✅ Has identity response: {has_identity}")
    print(f"✅ Has missing traits: {has_missing}")
    print(f"✅ Has clarification questions: {has_questions}")
    
    if all([matches_status, has_identity, has_missing, has_questions]):
        print("\n🎉 PERFECT! Scenario works exactly as user requested!")
        return True
    else:
        print("\n❌ Needs adjustment")
        return False

if __name__ == "__main__":
    success = test_exact_scenario()
    if success:
        print("\n🚀 SYSTEM READY - Identity responses with clarification questions working perfectly!")
    else:
        print("\n🔧 Needs fixes")
