#!/usr/bin/env python3
"""
Test the identity question issue fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_identity_question_issue():
    """Test the issue where identity questions reset missing traits."""
    
    print("Testing identity question trait preservation...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Test scenario 1: No identity question (baseline)
    print("\n=== Scenario 1: No Identity Question ===")
    result1 = analyzer.analyze(
        id=225985882206,
        user_input="مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى.",
        new_input=[],
        languages="ar"
    )
    
    print(f"Status: {result1['status']}")
    print(f"Missing traits: {result1['missing_traits']}")
    print(f"Description identity: {result1.get('description_identity', 'None')}")
    
    baseline_missing = set(result1['missing_traits'])
    print(f"Baseline missing traits: {baseline_missing}")
    
    # Test scenario 2: With identity question
    print("\n=== Scenario 2: With Identity Question ===")
    result2 = analyzer.analyze(
        id=225985882206,
        user_input="مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى.",
        new_input=[
            {
                "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
                "answer": "من انت"
            }
        ],
        languages="ar"
    )
    
    print(f"Status: {result2['status']}")
    print(f"Missing traits: {result2['missing_traits']}")
    print(f"Description identity: {result2.get('description_identity', 'None')}")
    
    identity_missing = set(result2['missing_traits'])
    print(f"Identity question missing traits: {identity_missing}")
    
    # Analysis
    print(f"\n=== Analysis ===")
    
    if result2.get('description_identity'):
        print("✅ Identity response provided correctly")
    else:
        print("❌ Identity response missing")
        
    # Check if traits were preserved
    if identity_missing <= baseline_missing:
        print("✅ Trait detection preserved or improved")
        print(f"   Baseline had {len(baseline_missing)} missing traits")
        print(f"   Identity scenario has {len(identity_missing)} missing traits")
        if len(identity_missing) < len(baseline_missing):
            print("   🎉 Actually improved trait detection!")
    else:
        print("❌ Trait detection regressed")
        print(f"   Baseline had {len(baseline_missing)} missing traits: {baseline_missing}")
        print(f"   Identity scenario has {len(identity_missing)} missing traits: {identity_missing}")
        extra_missing = identity_missing - baseline_missing
        print(f"   Extra missing traits: {extra_missing}")
        
    # Check the expected behavior
    print(f"\n=== Expected vs Actual ===")
    print("Expected behavior:")
    print("  - Identity response should be provided")
    print("  - Original user_input traits should be preserved")
    print("  - Only truly missing traits should be in missing_traits")
    
    print("Actual behavior:")
    print(f"  - Identity response: {'✅ Provided' if result2.get('description_identity') else '❌ Missing'}")
    print(f"  - Missing traits count: {len(identity_missing)} (baseline: {len(baseline_missing)})")
    
    if len(identity_missing) == len(baseline_missing) and result2.get('description_identity'):
        print("🎉 ISSUE FIXED: Identity question preserves trait detection!")
    elif len(identity_missing) < len(baseline_missing) and result2.get('description_identity'):
        print("🚀 BONUS: Identity question actually improved trait detection!")
    else:
        print("⚠️  Issue may still exist")

if __name__ == "__main__":
    test_identity_question_issue()
