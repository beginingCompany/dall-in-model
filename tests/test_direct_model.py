#!/usr/bin/env python3
"""
Direct test of the personality analyzer without API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_direct_identity_detection():
    """Test the analyzer directly"""
    
    analyzer = PersonalityAnalyzer()
    
    print("Testing Direct Model Identity Detection")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        "who are you",
        "مين مطورك", 
        "what is your purpose",
        "I am creative and analytical"
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        try:
            result = analyzer.analyze(
                id=i,
                user_input=user_input,
                languages="en" if not any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in user_input) else "ar"
            )
            
            identity = result.get("description_identity", "None")
            has_identity = "Yes" if identity != "None" and identity else "No"
            
            print(f"{i}. '{user_input}'")
            print(f"   Identity Response: {has_identity}")
            if has_identity == "Yes":
                print(f"   Response: {identity[:80]}...")
            print()
            
        except Exception as e:
            print(f"{i}. ERROR: {str(e)}")
    
    print("Direct test completed!")

if __name__ == "__main__":
    test_direct_identity_detection()
