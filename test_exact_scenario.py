#!/usr/bin/env python3
"""
Test the exact scenario reported by the user
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_exact_scenario():
    """Test the exact scenario from user report"""
    
    analyzer = PersonalityAnalyzer()
    
    # Exact input from user report
    result = analyzer.analyze(
        id=653876528763528675238678466547677578236,
        user_input="i am developer",
        new_input=[],
        languages="en"
    )
    
    print("Testing exact scenario: 'i am developer'")
    print("=" * 40)
    print(f"Input: 'i am developer'")
    print(f"ID: {result.get('id')}")
    print(f"Status: {result.get('status')}")
    print(f"Description Identity: {result.get('description_identity')}")
    print(f"Missing Traits: {result.get('missing_traits', [])}")
    print(f"Has Clarification Questions: {len(result.get('clarification_questions', []))} questions")
    
    # Check if it's correctly identified as personality (no identity response)
    has_identity = bool(result.get('description_identity'))
    
    if not has_identity:
        print("\n✅ SUCCESS: Correctly identified as personality description, not identity question")
        print("The system will now properly analyze this as personality traits.")
    else:
        print("\n❌ ISSUE: Still being detected as identity question")
        print(f"Identity Response: {result.get('description_identity')}")
    
    return not has_identity

if __name__ == "__main__":
    test_exact_scenario()
