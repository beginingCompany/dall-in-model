#!/usr/bin/env python3
"""
Test with a clear, complete personality scenario to see if GPT provides descriptions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_complete_personality_scenario():
    """Test with very clear complete personality data"""
    
    print("=== Testing Complete Personality Scenario ===\n")
    analyzer = PersonalityAnalyzer()
    
    # Clear, comprehensive personality data
    user_input = "Hello! I'm a data scientist who loves analytical thinking and problem-solving."
    
    new_input = [
        {
            "question": "How do you interact with others?",
            "answer": "I'm very social, love teamwork, and often take leadership roles in group projects."
        },
        {
            "question": "How do you handle emotions?", 
            "answer": "I stay calm under pressure, feel motivated by challenges, and manage stress through logical analysis."
        },
        {
            "question": "How do you approach problems?",
            "answer": "I think analytically, focus on details, make rational decisions, and love learning new concepts."
        },
        {
            "question": "What are your behavioral patterns?",
            "answer": "I'm very organized, follow consistent routines, meet deadlines reliably, and am highly disciplined in my work."
        }
    ]
    
    print("INPUT DATA:")
    print(f"User Input: {user_input}")
    print("Clear personality indicators:")
    for i, qa in enumerate(new_input, 1):
        print(f"  {i}. Q: {qa['question']}")
        print(f"     A: {qa['answer']}")
    
    print(f"\n{'='*60}")
    print("ANALYZING...")
    
    result = analyzer.analyze(
        id=225985882206,
        user_input=user_input,
        new_input=new_input,
        languages="en"
    )
    
    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    
    # Check the results
    status = result.get("status")
    description_english = result.get("description_english", "")
    description_identity = result.get("description_identity")
    missing_traits = result.get("missing_traits", [])
    
    print(f"Status: {status}")
    print(f"Missing traits: {missing_traits}")
    print(f"Identity response: {'YES' if description_identity else 'NO'}")
    print(f"English description length: {len(description_english)} chars")
    
    if status == "complete" and description_english and not description_identity:
        print("✅ PERFECT: Complete personality analysis with description")
        print(f"Description: {description_english[:100]}...")
    elif status == "complete" and description_identity:
        print("❌ ISSUE: Complete but showing identity instead of personality")
    elif status == "incomplete":
        print("❌ ISSUE: Should be complete with this much clear data")
    else:
        print("❌ UNEXPECTED result")

if __name__ == "__main__":
    test_complete_personality_scenario()
