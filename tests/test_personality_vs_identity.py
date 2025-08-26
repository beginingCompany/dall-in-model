#!/usr/bin/env python3
"""
Test to verify that personality statements are not confused with identity questions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_personality_vs_identity():
    """Test that personality statements are correctly distinguished from identity questions"""
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        # Personality statements (should NOT trigger identity response)
        {"input": "i am developer", "expected_identity": False, "description": "Self-description as developer"},
        {"input": "i am a team player", "expected_identity": False, "description": "Self-description about teamwork"},
        {"input": "i work in a team", "expected_identity": False, "description": "Work environment description"},
        {"input": "my role is to analyze data", "expected_identity": False, "description": "Personal role description"},
        {"input": "i have analytical thinking", "expected_identity": False, "description": "Cognitive trait description"},
        
        # Identity questions (should trigger identity response)
        {"input": "who are you", "expected_identity": True, "description": "Direct identity question"},
        {"input": "what is your role", "expected_identity": True, "description": "System role question"},
        {"input": "who is your developer", "expected_identity": True, "description": "Developer question"},
        {"input": "مين مطورك", "expected_identity": True, "description": "Arabic developer question"},
        {"input": "what do you do", "expected_identity": True, "description": "Function question"},
    ]
    
    print("Testing Personality vs Identity Detection")
    print("=" * 50)
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = analyzer.analyze(
                id=i,
                user_input=test_case["input"],
                languages="en" if not any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in test_case["input"]) else "ar"
            )
            
            identity_response = result.get("description_identity")
            has_identity = bool(identity_response)
            expected = test_case["expected_identity"]
            
            success = has_identity == expected
            success_count += success
            
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"{i:2d}. {status} | {test_case['description']}")
            print(f"     Input: '{test_case['input']}'")
            print(f"     Expected Identity: {expected} | Got: {has_identity}")
            
            if not success:
                print(f"     Identity Response: {identity_response}")
                print(f"     Status: {result.get('status')}")
                print(f"     Missing Traits: {result.get('missing_traits', [])}")
            
            print()
            
        except Exception as e:
            print(f"{i:2d}. ❌ ERROR | {test_case['description']}")
            print(f"     Error: {str(e)}")
            print()
    
    success_rate = (success_count / total_count) * 100
    print("=" * 50)
    print(f"RESULTS: {success_count}/{total_count} tests passed ({success_rate:.1f}%)")
    
    if success_count < total_count:
        print(f"\nFailed {total_count - success_count} tests - need to improve detection accuracy")
    else:
        print("\n🎉 All tests passed! Identity detection is working correctly.")
    
    return success_rate >= 90

if __name__ == "__main__":
    test_personality_vs_identity()
