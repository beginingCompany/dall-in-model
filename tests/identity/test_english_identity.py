#!/usr/bin/env python3

import sys
import json
sys.path.append('.')

from app.personality_analyzer import PersonalityAnalyzer

def test_english_identity():
    print("🧪 English Identity Test")
    print("=======================")
    
    analyzer = PersonalityAnalyzer()
    
    # Test English identity question
    test_input = "who are you"
    result = analyzer.analyze(id=126, user_input=test_input, new_input=[], languages="en")
    
    print(f"Input: '{test_input}'")
    print(f"description_identity: {repr(result['description_identity'])}")
    
    if result['description_identity'] and isinstance(result['description_identity'], str):
        print("✅ CORRECT: English identity response is non-empty string")
        print(f"Response: {result['description_identity']}")
    else:
        print("❌ ISSUE: Expected English string response")
    
    # Test English multi-question  
    print("\n" + "="*50)
    test_input2 = "who are you and what is your purpose"
    result2 = analyzer.analyze(id=127, user_input=test_input2, new_input=[], languages="en")
    
    print(f"Input: '{test_input2}'")
    print(f"description_identity: {repr(result2['description_identity'])}")
    
    if result2['description_identity'] and isinstance(result2['description_identity'], str):
        print("✅ CORRECT: English multi-identity response is non-empty string")
        print(f"Response: {result2['description_identity']}")
    else:
        print("❌ ISSUE: Expected English string response")

if __name__ == "__main__":
    test_english_identity()
