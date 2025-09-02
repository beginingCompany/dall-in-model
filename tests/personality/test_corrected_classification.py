#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_corrected_classification():
    """Test the corrected classification behavior"""
    print("🧪 Corrected Identity vs Personality Classification Test")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {
            "name": "Original User Problem - Should be Personality",
            "input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
            "should_have_identity": False,
            "description": "Pure personality description - should NOT trigger identity"
        },
        {
            "name": "Mixed Content - Should detect identity question",
            "input": "I'm a data scientist who loves patterns, but I want to know how you work",
            "should_have_identity": True,
            "description": "Mixed content with identity question - should trigger identity"
        },
        {
            "name": "Pure Identity Question",
            "input": "How do you analyze personalities?",
            "should_have_identity": True,
            "description": "Direct methodology question - should trigger identity"
        },
        {
            "name": "User Self-Description",
            "input": "I'm analytical and like to understand how things work",
            "should_have_identity": False,
            "description": "User describing their own analytical nature - should NOT trigger identity"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print(f"Expected: {test_case['description']}")
        print("-" * 50)
        
        result = analyzer.analyze(
            id=i,
            user_input=test_case['input'],
            new_input=[],
            languages="en"
        )
        
        has_identity = result.get('description_identity') is not None
        
        print(f"Identity Response: {'✅ YES' if has_identity else '❌ NO'}")
        print(f"Expected Identity: {'YES' if test_case['should_have_identity'] else 'NO'}")
        
        if has_identity == test_case['should_have_identity']:
            print("✅ PASS: Classification correct")
        else:
            print("❌ FAIL: Classification incorrect")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Classification working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED: Need further adjustment")
    
    return all_passed

if __name__ == "__main__":
    test_corrected_classification()
