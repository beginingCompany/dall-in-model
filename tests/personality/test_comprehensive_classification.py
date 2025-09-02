#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_comprehensive_classification():
    """Test various inputs to ensure correct personality vs identity classification"""
    print("🧪 Comprehensive Personality vs Identity Classification Test")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {
            "name": "Personality Description (should NOT trigger identity)",
            "input": "I enjoy working with data and solving analytical problems",
            "expected_identity": False,
            "expected_personality": True
        },
        {
            "name": "Identity Question (should trigger identity)",
            "input": "Who are you and what is your purpose?",
            "expected_identity": True,
            "expected_personality": False
        },
        {
            "name": "Methodology Question (should trigger identity)",
            "input": "How do you analyze personalities?",
            "expected_identity": True,
            "expected_personality": False
        },
        {
            "name": "User Analytical Nature (should NOT trigger identity)",
            "input": "I'm analytical and like to understand how things work",
            "expected_identity": False,
            "expected_personality": True
        },
        {
            "name": "Mixed Content with Identity Question",
            "input": "I'm a data scientist who loves patterns, but I want to know how you work",
            "expected_identity": True,
            "expected_personality": False  # Mixed content usually triggers identity
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print("-" * 50)
        
        result = analyzer.analyze(
            id=i,
            user_input=test_case['input'],
            new_input=[],
            languages="en"
        )
        
        has_identity = result.get('description_identity') is not None
        has_personality = result.get('description_english') != "" or result.get('status') == 'incomplete'
        
        print(f"Identity Response: {'✅ YES' if has_identity else '❌ NO'}")
        print(f"Personality Analysis: {'✅ YES' if has_personality else '❌ NO'}")
        
        # Check expectations
        identity_correct = has_identity == test_case['expected_identity']
        personality_correct = has_personality == test_case['expected_personality']
        
        if identity_correct and personality_correct:
            print("✅ PASS: Classification correct")
        else:
            print("❌ FAIL: Classification incorrect")
            if not identity_correct:
                print(f"   Expected identity: {test_case['expected_identity']}, Got: {has_identity}")
            if not personality_correct:
                print(f"   Expected personality: {test_case['expected_personality']}, Got: {has_personality}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Identity vs Personality classification working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED: Classification needs further refinement")
    
    return all_passed

if __name__ == "__main__":
    test_comprehensive_classification()
