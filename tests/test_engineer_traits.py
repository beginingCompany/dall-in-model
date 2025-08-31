#!/usr/bin/env python3
"""
Test if GPT extracts engineer job as personality trait
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_engineer_trait_extraction():
    """Test if engineer job gets extracted as personality trait"""
    
    print("🧪 TESTING ENGINEER TRAIT EXTRACTION")
    print("=" * 45)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case: User introduces as engineer
    test_case = {
        "id": 12345,
        "user_input": "انا المهندس احمد",
        "new_input": [],
        "languages": "ar"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print(f"Question: Will GPT extract 'engineer' as personality trait?")
    print("-" * 45)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: '{response.get('description_arabic', '')}'")
    print(f"English Description: '{response.get('description_english', '')}'")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    
    # Check if descriptions mention engineering/analytical traits
    arabic_desc = response.get('description_arabic', '')
    english_desc = response.get('description_english', '')
    
    engineering_keywords = ['مهندس', 'هندسة', 'تحليل', 'engineer', 'engineering', 'analytical', 'technical']
    
    found_engineering_traits = []
    for keyword in engineering_keywords:
        if keyword in arabic_desc.lower() or keyword in english_desc.lower():
            found_engineering_traits.append(keyword)
    
    print(f"\nEngineering-related words found: {found_engineering_traits}")
    
    if found_engineering_traits:
        print("✅ YES: GPT extracted engineering/analytical traits from job title")
    else:
        print("❌ NO: GPT did not extract personality traits from 'engineer' job title")
    
    # Check if all traits are covered
    missing_traits = response.get('missing_traits', [])
    if not missing_traits:
        print("✅ All traits covered from just the job title!")
    else:
        print(f"⚠️  Still missing traits: {missing_traits}")
        print("   GPT needs more information beyond just job title")
    
    print("\n" + "=" * 45)

if __name__ == "__main__":
    test_engineer_trait_extraction()
