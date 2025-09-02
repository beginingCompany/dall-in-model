#!/usr/bin/env python3
"""
Test engineer who struggles with problem-solving (contradicts professional stereotypes)
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_engineer_contradicting_stereotypes():
    """Test engineer who doesn't fit typical analytical stereotype"""
    
    print("🧪 TESTING ENGINEER WHO CONTRADICTS STEREOTYPES")
    print("=" * 55)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case: Engineer who struggles with problem-solving
    test_case = {
        "id": 12345,
        "user_input": "انا المهندس احمد لكن اجد صعوبة في حل المشاكل المعقدة واشعر بالتوتر الشديد واحتاج مساعدة في اتخاذ القرارات",
        "new_input": [],
        "languages": "ar"
    }
    
    print(f"Testing input: '{test_case['user_input']}'")
    print("Translation: 'I am engineer Ahmed but I find difficulty in solving complex problems and feel severe stress and need help in decision making'")
    print("Expected: System should trust self-description over job stereotypes")
    print("-" * 55)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: '{response.get('description_arabic', '')}'")
    print(f"English Description: '{response.get('description_english', '')}'")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    
    # Check if descriptions reflect actual behavior vs job stereotypes
    arabic_desc = response.get('description_arabic', '')
    english_desc = response.get('description_english', '')
    
    # Look for problem-solving struggle vs analytical strength
    struggle_keywords = ['صعوبة', 'توتر', 'مساعدة', 'difficulty', 'stress', 'help', 'struggle']
    analytical_keywords = ['تحليل', 'منطقي', 'منهجي', 'analytical', 'logical', 'systematic']
    
    struggle_found = any(keyword in arabic_desc.lower() or keyword in english_desc.lower() for keyword in struggle_keywords)
    analytical_assumed = any(keyword in arabic_desc.lower() or keyword in english_desc.lower() for keyword in analytical_keywords)
    
    print(f"\nDescription Analysis:")
    print(f"  Reflects actual struggles: {'✅ YES' if struggle_found else '❌ NO'}")
    print(f"  Assumes analytical from job: {'⚠️ YES (stereotyping)' if analytical_assumed else '✅ NO (good)'}")
    
    if struggle_found and not analytical_assumed:
        print("\n✅ EXCELLENT: System trusts self-description over job stereotypes!")
        print("   Prioritizes actual behavior over professional assumptions")
    elif struggle_found and analytical_assumed:
        print("\n⚠️ MIXED: System acknowledges struggles but still assumes analytical traits")
    else:
        print("\n❌ POOR: System ignores self-description and assumes job stereotypes")
    
    print("\n" + "=" * 55)

if __name__ == "__main__":
    test_engineer_contradicting_stereotypes()
