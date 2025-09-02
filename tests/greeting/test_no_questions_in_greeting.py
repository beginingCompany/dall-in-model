#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_arabic_greeting_no_questions():
    """Test that Arabic greetings don't include questions like 'كيف يمكنني مساعدتك اليوم؟'"""
    print("🧪 Testing Arabic Greeting Response (No Questions)")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Test case from user's example
    test_input = "مرحبا! انا احمد كيف حالك؟"
    
    print(f"Input: {test_input}")
    print("-" * 60)
    
    # Test the analysis
    result = analyzer.analyze(
        id=102,
        user_input=test_input,
        new_input=[],
        languages="ar"
    )
    
    print("Result:")
    print(f"Status: {result['status']}")
    print(f"Greeting Response: {result['personal_greeting_and_off_topic']}")
    print(f"Identity Response: {result['description_identity']}")
    print()
    
    # Check if the greeting response contains questions
    greeting_response = result['personal_greeting_and_off_topic']
    contains_question = '؟' in greeting_response or 'كيف يمكنني' in greeting_response
    
    if contains_question:
        print("❌ FAIL: Greeting response contains questions")
        print(f"   Found question marks or 'كيف يمكنني' in: {greeting_response}")
        return False
    else:
        print("✅ PASS: Greeting response does not contain questions")
        print(f"   Clean response: {greeting_response}")
        return True

if __name__ == "__main__":
    test_arabic_greeting_no_questions()
