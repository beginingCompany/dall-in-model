#!/usr/bin/env python3
"""
Test if user's name appears in personality descriptions
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_name_in_description():
    """Test if user's name appears in personality descriptions"""
    
    print("🧪 TESTING NAME IN PERSONALITY DESCRIPTIONS")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Test with comprehensive input to get complete status
    test_case = {
        "id": 12345,
        "user_input": "انا المهندس احمد احب العمل مع الفرق واحلل البيانات بهدوء واتخذ القرارات بعقلانية واتحكم في مشاعري بصبر",
        "new_input": [],
        "languages": "ar"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print("Expected: Name 'احمد' should appear in descriptions")
    print("-" * 50)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: '{response.get('description_arabic', '')}'")
    print(f"English Description: '{response.get('description_english', '')}'")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    
    # Check if name appears in descriptions
    arabic_desc = response.get('description_arabic', '')
    english_desc = response.get('description_english', '')
    
    name_in_arabic = 'احمد' in arabic_desc
    name_in_english = 'احمد' in english_desc or 'Ahmed' in english_desc
    
    print("\nName Detection Results:")
    print(f"  Name in Arabic description: {'✅ YES' if name_in_arabic else '❌ NO'}")
    print(f"  Name in English description: {'✅ YES' if name_in_english else '❌ NO'}")
    
    if name_in_arabic or name_in_english:
        print("\n✅ SUCCESS: User's name appears in personality descriptions!")
    else:
        print("\n❌ FAILED: User's name is missing from personality descriptions")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_name_in_description()
