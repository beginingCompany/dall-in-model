#!/usr/bin/env python3
"""
Test personal greeting fix for Arabic introduction
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_personal_greeting():
    """Test personal greeting for Arabic introduction"""
    
    print("🧪 TESTING PERSONAL GREETING FIX")
    print("=" * 40)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case: Arabic introduction that should generate greeting
    test_case = {
        "id": 225985882206,
        "user_input": "انا المهندس احمد",
        "new_input": [],
        "languages": "ar"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print(f"Expected: Should detect name 'احمد' and job 'مهندس' and create Arabic greeting")
    print("-" * 40)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: {response.get('description_arabic', '')}")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    print(f"Questions: {len(response.get('clarification_questions', []))}")
    
    # Check if personal greeting exists and is not empty
    greeting = response.get('personal_greeting', '')
    if greeting and len(greeting.strip()) > 0:
        print("\n✅ SUCCESS: Personal greeting generated!")
        print(f"   Greeting: {greeting}")
    else:
        print("\n❌ FAILED: Personal greeting is empty")
        
    print("\n" + "=" * 40)

if __name__ == "__main__":
    test_personal_greeting()
