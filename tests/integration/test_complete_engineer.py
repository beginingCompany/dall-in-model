#!/usr/bin/env python3
"""
Test complete engineer profile to see traits in final descriptions
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_complete_engineer_profile():
    """Test complete engineer profile to see all traits including job-based ones"""
    
    print("🧪 TESTING COMPLETE ENGINEER PROFILE")
    print("=" * 45)
    
    analyzer = PersonalityAnalyzer()
    
    # Complete engineer profile with all 4 trait types
    test_case = {
        "id": 12345,
        "user_input": "انا المهندس احمد احب العمل مع الفرق واحلل المشاكل بمنهجية واتحكم في مشاعري وأبقى هادئا تحت الضغط",
        "new_input": [],
        "languages": "ar"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print("(Complete: Engineer + Social + Cognitive + Behavioral + Emotional)")
    print("Expected: Complete status with engineering traits in descriptions")
    print("-" * 45)
    
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
        'مهندس', 'تحليل', 'منهجية', 'منطقي',  # Arabic
        'engineer', 'analytical', 'methodical', 'systematic', 'logical'  # English
    ]
    
    found_traits = []
    for keyword in engineering_keywords:
        if keyword in arabic_desc.lower() or keyword in english_desc.lower():
            found_traits.append(keyword)
    
    print(f"\nEngineering-related terms in descriptions: {found_traits}")
    
    if response.get('status') == 'complete':
        print("✅ COMPLETE: All traits covered including engineer-based traits!")
        if found_traits:
            print("✅ TRAITS VISIBLE: Engineering characteristics appear in final descriptions!")
        else:
            print("⚠️  TRAITS HIDDEN: Engineering traits covered but not explicitly mentioned")
    else:
        print("❌ INCOMPLETE: Still missing some traits")
    
    print("\n" + "=" * 45)

if __name__ == "__main__":
    test_complete_engineer_profile()
