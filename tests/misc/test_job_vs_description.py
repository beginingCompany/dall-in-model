#!/usr/bin/env python3
"""
Compare job title vs detailed description for trait extraction
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_job_vs_description():
    """Compare job title vs detailed description"""
    
    print("🧪 COMPARING JOB TITLE VS DETAILED DESCRIPTION")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # Test 1: Just job title
    test1 = {
        "id": 12345,
        "user_input": "انا المهندس احمد",
        "new_input": [],
        "languages": "ar"
    }
    
    # Test 2: Detailed engineering description
    test2 = {
        "id": 67890,
        "user_input": "انا مهندس احب حل المشاكل المعقدة واحلل البيانات بدقة واعمل بمنهجية منظمة",
        "new_input": [],
        "languages": "ar"
    }
    
    print("TEST 1: Just job title")
    print(f"Input: '{test1['user_input']}'")
    result1 = analyzer.analyze(**test1)
    response1 = json.loads(result1["content"])
    print(f"Status: {response1.get('status')}")
    print(f"Missing Traits: {response1.get('missing_traits', [])}")
    print(f"Arabic Description: '{response1.get('description_arabic', '')}'")
    
    print("\n" + "-" * 50)
    
    print("TEST 2: Detailed description")
    print(f"Input: '{test2['user_input']}'")
    result2 = analyzer.analyze(**test2)
    response2 = json.loads(result2["content"])
    print(f"Status: {response2.get('status')}")
    print(f"Missing Traits: {response2.get('missing_traits', [])}")
    print(f"Arabic Description: '{response2.get('description_arabic', '')}'")
    
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    
    if len(response1.get('missing_traits', [])) > len(response2.get('missing_traits', [])):
        print("✅ Detailed description extracts more traits than job title")
        print("   Job titles are for greetings, descriptions are for personality analysis")
    else:
        print("⚠️  Both give similar results")

if __name__ == "__main__":
    test_job_vs_description()
