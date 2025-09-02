#!/usr/bin/env python3
"""
Test trait extraction from engineer job title with more context
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_engineer_with_social_context():
    """Test engineer job title with some social context to get descriptions"""
    
    print("🧪 TESTING ENGINEER TRAITS WITH SOCIAL CONTEXT")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Test: Engineer + minimal social context to get descriptions
    test_case = {
        "id": 12345,
        "user_input": "انا المهندس احمد احب العمل مع الفرق واشعر بالفخر عندما احل المشاكل",
        "new_input": [],
        "languages": "ar"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print("(Engineer + team work + pride in problem solving)")
    print("Expected: Should extract cognitive/behavioral traits from 'engineer'")
    print("-" * 50)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: '{response.get('description_arabic', '')}'")
    print(f"English Description: '{response.get('description_english', '')}'")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    
    # Check for engineering-related traits in descriptions
    arabic_desc = response.get('description_arabic', '')
    english_desc = response.get('description_english', '')
    
    engineering_keywords = [
        'تحليل', 'حل المشاكل', 'منطقي', 'منهجي', 'دقيق',  # Arabic
        'analytical', 'problem-solving', 'logical', 'systematic', 'methodical', 'engineer'  # English
    ]
    
    found_traits = []
    for keyword in engineering_keywords:
        if keyword in arabic_desc.lower() or keyword in english_desc.lower():
            found_traits.append(keyword)
    
    print(f"\nEngineering traits found in descriptions: {found_traits}")
    
    if found_traits:
        print("✅ SUCCESS: GPT extracted engineering-related personality traits!")
        print(f"   Traits: {', '.join(found_traits)}")
    else:
        print("❌ FAILED: No engineering traits found in descriptions")
    
    # Check if cognitive/behavioral traits are no longer missing
    missing_traits = response.get('missing_traits', [])
    cognitive_missing = 'cognitive' in missing_traits
    behavioral_missing = 'behavioral' in missing_traits
    
    print(f"\nTrait Coverage Analysis:")
    print(f"  Cognitive missing: {'❌ YES' if cognitive_missing else '✅ NO'}")
    print(f"  Behavioral missing: {'❌ YES' if behavioral_missing else '✅ NO'}")
    
    if not cognitive_missing and not behavioral_missing:
        print("✅ EXCELLENT: Engineer job provided cognitive + behavioral traits!")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_engineer_with_social_context()
