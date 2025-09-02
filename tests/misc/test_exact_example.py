#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_exact_user_example():
    """Test the exact example from the user to confirm the fix"""
    print("🧪 Testing User's Exact Example")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # User's exact input
    result = analyzer.analyze(
        id=102,
        user_input="مرحبا! انا احمد كيف حالك؟",
        new_input=[],
        languages="ar"
    )
    
    print("📥 INPUT:")
    print(json.dumps({
        "id": 102,
        "user_input": "مرحبا! انا احمد كيف حالك؟",
        "new_input": [],
        "languages": "ar"
    }, ensure_ascii=False, indent=2))
    
    print("\n📤 OUTPUT:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Verify no questions in greeting response
    greeting = result['personal_greeting_and_off_topic']
    contains_question = '؟' in greeting or 'كيف يمكنني' in greeting
    
    print(f"\n🔍 ANALYSIS:")
    print(f"Greeting contains questions: {'❌ YES' if contains_question else '✅ NO'}")
    print(f"Identity field is null: {'✅ YES' if result['description_identity'] is None else '❌ NO'}")
    print(f"Status: {result['status']}")
    
    if not contains_question and result['description_identity'] is None:
        print("\n🎉 SUCCESS: Perfect response format!")
        return True
    else:
        print("\n❌ ISSUE: Response format needs adjustment")
        return False

if __name__ == "__main__":
    test_exact_user_example()
