#!/usr/bin/env python3
"""
Test engineer trait extraction with forced completion
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_engineer_with_completion():
    """Test engineer trait extraction with enough info to complete"""
    
    print("🧪 TESTING ENGINEER TRAITS WITH COMPLETION")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case: Engineer + minimal social/emotional info to trigger completion
    test_case = {
        "id": 12345,
        "user_input": "انا المهندس احمد احب التعامل مع الناس واشعر بالهدوء",
        "new_input": [],
        "languages": "ar"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print("Engineer + social + emotional hints to trigger completion")
    print("-" * 50)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: '{response.get('description_arabic', '')}'")
    print(f"English Description: '{response.get('description_english', '')}'")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    
    # Check for engineering-related traits
    arabic_desc = response.get('description_arabic', '')
    english_desc = response.get('description_english', '')
    
    engineering_keywords = ['تحليل', 'منهجي', 'مهندس', 'analytical', 'methodical', 'engineer', 'problem', 'technical']
    
    found_traits = []
    for keyword in engineering_keywords:
        if keyword in arabic_desc.lower() or keyword in english_desc.lower():
            found_traits.append(keyword)
    
    print(f"\nEngineering traits found: {found_traits}")
    
    if found_traits:
        print("✅ SUCCESS: GPT extracted engineering traits!")
        print("   Job title successfully influenced personality analysis")
    else:
        print("❌ FAILED: No engineering traits found in descriptions")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_engineer_with_completion()
