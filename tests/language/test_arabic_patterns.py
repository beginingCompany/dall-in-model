#!/usr/bin/env python3
"""
Test Arabic pattern matching specifically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_arabic_patterns():
    """Test Arabic pattern matching."""
    
    print("Testing Arabic pattern matching...")
    
    # Test Arabic text from the log
    arabic_text1 = "أحب العمل ضمن الفرق وغالبًا ما أجد نفسي أتولى أدوار القيادة بشكل طبيعي. أستمتع بإرشاد الزملاء الجدد وتيسير المناقشات الجماعية."
    arabic_text2 = "مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى."
    
    print(f"\nText 1: {arabic_text1}")
    traits1 = PersonalityAnalyzer._extract_traits_by_pattern(arabic_text1)
    print(f"Pattern-detected traits: {traits1}")
    print("Expected: emotional (أحب, أستمتع), social (الفرق, القيادة, إرشاد, الزملاء), behavioral (أتولى أدوار)")
    
    print(f"\nText 2: {arabic_text2}")
    traits2 = PersonalityAnalyzer._extract_traits_by_pattern(arabic_text2)
    print(f"Pattern-detected traits: {traits2}")
    print("Expected: emotional (يستمتع, أشعر برضا), cognitive (البيانات, المشكلات التحليلية, الأنماط)")
    
    combined_traits = traits1.union(traits2)
    print(f"\nCombined traits: {combined_traits}")
    missing_traits = {"emotional", "social", "cognitive", "behavioral"} - combined_traits
    print(f"Missing traits: {missing_traits}")
    
    if len(missing_traits) <= 1:
        print("✅ Arabic pattern matching is working well!")
    else:
        print("⚠️  Arabic pattern matching needs improvement")
        
    # Test the full analyzer with Arabic
    print(f"\n=== Testing Full Arabic Analysis ===")
    analyzer = PersonalityAnalyzer()
    
    # Simulate the exact scenario from the log
    result = analyzer.analyze(
        id=225985882206,
        user_input=arabic_text2 + " " + arabic_text1,  # Combined text
        new_input=[{
            'question': 'كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟',
            'answer': arabic_text1
        }],
        languages='ar'
    )
    
    print(f"Status: {result['status']}")
    print(f"Missing traits: {result['missing_traits']}")
    print(f"Clarification questions: {result['clarification_questions']}")
    
    if len(result['missing_traits']) <= 1:
        print("✅ Full Arabic analysis improved!")
    else:
        print("⚠️  Full Arabic analysis still has issues")

if __name__ == "__main__":
    test_arabic_patterns()
