#!/usr/bin/env python3
"""
Test the full analyze method with the exact user request data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_full_analyze_method():
    """Test the full analyze method with exact user request data"""
    
    print("=== Testing Full Analyze Method ===")
    analyzer = PersonalityAnalyzer()
    
    # Exact data from user's request
    result = analyzer.analyze(
        id=225985882206,
        user_input="who are you",
        new_input=[
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            }
        ],
        languages="en"
    )
    
    print("RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    
    if result.get("description_identity"):
        print(f"❌ IDENTITY TRIGGERED: {result['description_identity']}")
        print("❌ This should NOT happen - 'who are you' is a confused answer in conversation")
        return False
    else:
        print("✅ NO IDENTITY RESPONSE: Correct behavior")
        return True

if __name__ == "__main__":
    print("Testing the full analyze method with exact user request data...\n")
    
    test_passed = test_full_analyze_method()
    
    print("\n" + "="*60)
    if test_passed:
        print("🎉 SUCCESS: The analyze method is working correctly!")
        print("✅ No identity response triggered for confused answer in conversation")
    else:
        print("❌ ISSUE FOUND: The analyze method is incorrectly triggering identity responses")
        print("🔧 Need to investigate the analyze method logic")
