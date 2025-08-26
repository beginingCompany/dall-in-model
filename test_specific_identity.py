#!/usr/bin/env python3
"""
Test specific identity question detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_specific_identity():
    """Test specific identity questions"""
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        "what is your role",
        "who are you", 
        "i am developer",
    ]
    
    for i, input_text in enumerate(test_cases, 1):
        result = analyzer.analyze(
            id=i,
            user_input=input_text,
            languages="en"
        )
        
        identity = result.get("description_identity")
        print(f"{i}. '{input_text}'")
        print(f"   Identity: {'Yes' if identity else 'No'}")
        if identity:
            print(f"   Response: {identity[:60]}...")
        print()

if __name__ == "__main__":
    test_specific_identity()
