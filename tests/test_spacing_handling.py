#!/usr/bin/env python3
"""
Test to verify that spacing in input is handled correctly with strip()
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_spacing_handling():
    """Test that input spacing is handled correctly"""
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {"input": "who are you", "description": "No extra spaces"},
        {"input": "  who are you  ", "description": "Leading and trailing spaces"},
        {"input": "   مين مطورك   ", "description": "Arabic with spaces"},
        {"input": "\twho are you\n", "description": "Tabs and newlines"},
        {"input": "  I am a creative person  ", "description": "Personality with spaces"},
    ]
    
    print("Testing Input Spacing Handling")
    print("=" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = analyzer.analyze(
                id=i,
                user_input=test_case["input"],
                languages="en" if not any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in test_case["input"]) else "ar"
            )
            
            identity_response = result.get("description_identity")
            status = result.get("status")
            
            print(f"{i}. {test_case['description']}")
            print(f"   Input: '{test_case['input']}'")
            print(f"   Status: {status}")
            print(f"   Has Identity: {'Yes' if identity_response else 'No'}")
            
            if identity_response:
                print(f"   Identity: {identity_response[:50]}...")
            
            print("   ✅ Processed successfully")
            print()
            
        except Exception as e:
            print(f"{i}. ERROR with '{test_case['input']}': {str(e)}")
            print()
    
    print("Spacing handling test completed!")

if __name__ == "__main__":
    test_spacing_handling()
