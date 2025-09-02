#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_weather_greeting():
    """Test the specific case that had a question mark"""
    print("🧪 Testing Weather Greeting (No Question Marks)")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    test_input = "Good morning! Nice weather today."
    
    print(f"Input: {test_input}")
    print("-" * 50)
    
    result = analyzer.analyze(
        id=999,
        user_input=test_input,
        new_input=[],
        languages="en"
    )
    
    greeting_response = result['personal_greeting_and_off_topic']
    print(f"Response: {greeting_response}")
    print()
    
    # Check for any question marks
    has_question_mark = '?' in greeting_response or '؟' in greeting_response
    
    if has_question_mark:
        print("❌ FAIL: Response contains question marks")
        print(f"   Found in: {greeting_response}")
        return False
    else:
        print("✅ PASS: No question marks found")
        return True

if __name__ == "__main__":
    test_weather_greeting()
