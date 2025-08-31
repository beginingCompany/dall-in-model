#!/usr/bin/env python3
"""
Test the exact same content but in English to see if it's a language issue
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_engineer_contradictions_english():
    """Test engineer contradictions in English"""
    
    print("🧪 TESTING ENGINEER CONTRADICTIONS IN ENGLISH")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Same content in English
    test_case = {
        "id": 12345,
        "user_input": "I am engineer Ahmed but I find difficulty in solving complex problems and feel severe stress and need help in decision making",
        "new_input": [],
        "languages": "en"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print("Expected: Should extract struggles as actual personality traits")
    print("-" * 50)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: '{response.get('description_arabic', '')}'")
    print(f"English Description: '{response.get('description_english', '')}'")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    
    # Check content
    english_desc = response.get('description_english', '')
    struggle_keywords = ['difficulty', 'stress', 'help', 'struggle', 'challenge']
    analytical_keywords = ['analytical', 'logical', 'systematic']
    
    struggle_found = any(keyword in english_desc.lower() for keyword in struggle_keywords)
    analytical_assumed = any(keyword in english_desc.lower() for keyword in analytical_keywords)
    
    print(f"\nDescription Analysis:")
    print(f"  Reflects actual struggles: {'✅ YES' if struggle_found else '❌ NO'}")
    print(f"  Assumes analytical from job: {'⚠️ YES (stereotyping)' if analytical_assumed else '✅ NO (good)'}")
    
    if struggle_found and not analytical_assumed:
        print("\n✅ EXCELLENT: System trusts self-description over job stereotypes!")
    elif not struggle_found and len(response.get('missing_traits', [])) < 4:
        print("\n⚠️ INCOMPLETE BUT IMPROVED: System extracted some traits from context")
    else:
        print("\n❌ POOR: System not extracting behavioral information properly")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_engineer_contradictions_english()
