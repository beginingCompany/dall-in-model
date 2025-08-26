#!/usr/bin/env python3
"""
Quick test for the developer trigger fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_developer_triggers():
    # Import the method directly
    from personality_analyzer import PersonalityAnalyzer
    
    test_cases = [
        "who is ur developer",
        "who is your developer", 
        "ur developer",
        "who's ur developer",
        "who made you",
        "مين مطورك"
    ]
    
    print("Testing developer trigger variations...\n")
    
    for i, test_input in enumerate(test_cases, 1):
        response = PersonalityAnalyzer.get_identity_response(test_input, "en")
        print(f"Test {i}: '{test_input}'")
        if response:
            print(f"✅ MATCHED: {response[:60]}...")
        else:
            print(f"❌ NO MATCH")
        print()

if __name__ == "__main__":
    test_developer_triggers()
