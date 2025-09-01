#!/usr/bin/env python3
"""
Test the exact scenario to see why identity response is showing instead of personality description when complete.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_complete_scenario():
    """Test the scenario where analysis should be complete with personality description"""
    
    print("=== Testing Complete Scenario ===")
    analyzer = PersonalityAnalyzer()
    
    # The exact user scenario
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "who are you"  # This should be ignored as confused answer
        },
        {
            "question": "How do you typically approach and handle complex problem-solving tasks?",
            "answer": "i analytical can solving the problems by analyze them"
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "i analytical can solving the problems by analyze them"
        }
    ]
    
    print(f"User Input: '{user_input}'")
    print(f"Conversation: {len(new_input)} Q&A exchanges")
    print("Expected: Complete personality description, NOT identity response")
    
    result = analyzer.analyze(
        id=225985882206,
        user_input=user_input,
        new_input=new_input,
        languages="en"
    )
    
    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Analysis
    print(f"\n{'='*50}")
    print("ANALYSIS:")
    
    if result.get("description_identity"):
        print(f"❌ PROBLEM: Identity response triggered: {result['description_identity']}")
        print("❌ This should provide personality description instead!")
        
    if result.get("description_english") and len(result["description_english"]) > 50:
        print(f"✅ GOOD: Personality description provided")
        print(f"   Description: {result['description_english'][:100]}...")
        
    if result.get("status") == "complete":
        print(f"✅ Status: Complete")
    else:
        print(f"⚠️ Status: {result.get('status')} - Missing: {result.get('missing_traits')}")
        
    return result

if __name__ == "__main__":
    test_complete_scenario()
