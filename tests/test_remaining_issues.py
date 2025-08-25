#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_remaining_issues():
    """Test the specific remaining issues"""
    
    print("=" * 60)
    print("TESTING REMAINING ISSUES")
    print("=" * 60)
    
    # Test the problematic single words
    problem_tests = [
        ("role", "Should NOT trigger (too generic)"),
        ("function", "Should NOT trigger (too generic)"),
        ("team", "Should NOT trigger (too generic)"),
        ("objectives", "Should NOT trigger (too generic)"),
        ("goals", "Should NOT trigger (too generic)"),
        ("What is your role?", "Should trigger"),
        ("Who is your team?", "Should trigger"),
        ("What are the objectives of BEGINING?", "Should trigger"),
        ("I work as a team leader", "Should NOT trigger"),
        ("My role in the company", "Should NOT trigger"),
        ("The objectives are clear", "Should NOT trigger"),
    ]
    
    for test_input, expected in problem_tests:
        response = PersonalityAnalyzer.get_identity_response(test_input, "en")
        
        should_trigger = "Should trigger" in expected
        has_response = response != ""
        
        if should_trigger == has_response:
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} '{test_input}' -> {expected}")
        if has_response:
            print(f"    Response: {response[:50]}...")
        else:
            print(f"    No response")
        print()

if __name__ == "__main__":
    test_remaining_issues()
