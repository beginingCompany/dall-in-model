#!/usr/bin/env python3
"""
Test the mixed content fix for the specific failing scenario
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_mixed_content_fix():
    print("Testing mixed content fix...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # The problematic input from the log (user_input with identity question appended)
    contaminated_input = "مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى. من انت"
    
    print(f"Contaminated input: {contaminated_input}")
    print("=" * 80)
    
    # Test the new cleaning method
    cleaned = analyzer._extract_personality_content_from_mixed_input(contaminated_input, "ar")
    print(f"Cleaned personality content: {cleaned}")
    print(f"Length of cleaned content: {len(cleaned.strip())}")
    print("=" * 80)
    
    # Test the full analysis with the corrected new_input format
    new_input = [
        {
            "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
            "answer": "من انت"
        }
    ]
    
    # Use only the personality description (without the identity question)
    clean_user_input = "مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى."
    
    print(f"Clean user input: {clean_user_input}")
    print(f"New input: {new_input}")
    print("=" * 80)
    
    # Run analysis
    result = analyzer.analyze(
        id=225985882206,
        user_input=clean_user_input,
        new_input=new_input,
        languages="ar"
    )
    
    print("\n=== ANALYSIS RESULT ===")
    print(f"Status: {result.get('status')}")
    print(f"Missing traits: {result.get('missing_traits')}")
    print(f"Number missing: {len(result.get('missing_traits', []))}")
    print(f"Identity description: {result.get('description_identity', '')}")
    print(f"Arabic description: {result.get('description_arabic', '')}")
    
    expected_missing = 1  # Only social should be missing
    actual_missing = len(result.get('missing_traits', []))
    
    print(f"\n=== COMPARISON ===")
    print(f"Expected missing traits: 1 (social)")
    print(f"Actual missing traits: {actual_missing}")
    print(f"User reported: 4 missing (all traits)")
    
    if actual_missing == 4:
        print("❌ STILL BROKEN: All traits missing - cleaning didn't work")
    elif actual_missing == 1:
        print("✅ FIXED: Only 1 trait missing as expected")
    elif actual_missing == 0:
        print("✅ EXCELLENT: All traits detected")
    else:
        print(f"⚠️ PARTIAL: {actual_missing} traits missing")
    
    return result

if __name__ == "__main__":
    test_mixed_content_fix()
