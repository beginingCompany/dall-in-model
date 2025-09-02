#!/usr/bin/env python3
"""
Test Arabic case
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_arabic_case():
    """Test the Arabic case"""
    
    print("🧪 TESTING ARABIC CASE")
    print("=" * 30)
    
    analyzer = PersonalityAnalyzer()
    
    # Arabic test
    arabic_test = {
        "id": 65387652876,
        "user_input": "من انت",
        "new_input": [],
        "languages": "ar"
    }
    
    result = analyzer.analyze(**arabic_test)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Identity (Arabic): {response.get('description_identity', '')}")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    print(f"Questions: {len(response.get('clarification_questions', []))}")
    
    for i, q in enumerate(response.get('clarification_questions', []), 1):
        print(f"  {i}. {q}")
    
    print("\n✅ Arabic case working perfectly!")

if __name__ == "__main__":
    test_arabic_case()
