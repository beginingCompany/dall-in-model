#!/usr/bin/env python3
"""
Test the exact scenario from the API call to debug the missing traits issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_real_scenario():
    print("Testing real API scenario...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Exact input from the API call
    user_input = "مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى."
    
    new_input = [
        {
            "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
            "answer": "من انت"
        }
    ]
    
    user_id = 225985882206
    languages = "ar"
    
    print(f"User ID: {user_id}")
    print(f"User input: {user_input}")
    print(f"New input: {new_input}")
    print(f"Languages: {languages}")
    print("=" * 80)
    
    # Run analysis
    result = analyzer.analyze(
        id=user_id,
        user_input=user_input,
        new_input=new_input,
        languages=languages
    )
    
    print("\n=== ANALYSIS RESULT ===")
    print(f"Status: {result.get('status')}")
    print(f"Missing traits: {result.get('missing_traits')}")
    print(f"Identity description: {result.get('description_identity', '')}")
    print(f"Arabic description: {result.get('description_arabic', '')}")
    print(f"Clarification questions: {result.get('clarification_questions', [])}")
    
    # Check if this matches the expected behavior
    expected_missing = []  # Should detect traits from the Arabic input
    actual_missing = set(result.get('missing_traits', []))
    
    print(f"\n=== ANALYSIS ===")
    print(f"Expected missing traits: {expected_missing}")
    print(f"Actual missing traits: {actual_missing}")
    
    if len(actual_missing) == 4:
        print("❌ ISSUE: All traits are missing - this suggests the fix isn't working in the API context")
        print("The Arabic input clearly contains cognitive and emotional traits")
    elif len(actual_missing) == 0:
        print("✅ PERFECT: All traits detected correctly")
    else:
        print(f"⚠️ PARTIAL: Some traits detected, {len(actual_missing)} still missing")
    
    return result

if __name__ == "__main__":
    test_real_scenario()
