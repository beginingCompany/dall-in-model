#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_title_recognition():
    """Test recognition of professional titles with names"""
    print("🎓 Testing Title + Name Recognition")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # User's exact input with title
    test_input = "مرحبا! انا المهندس احمد كيف حالك؟"
    
    print(f"📝 INPUT: {test_input}")
    print(f"   Translation: 'Hello! I am Engineer Ahmed, how are you?'")
    print("-" * 50)
    
    result = analyzer.analyze(
        id=102,
        user_input=test_input,
        new_input=[],
        languages="ar"
    )
    
    greeting = result['personal_greeting_and_off_topic']
    print(f"📤 RESPONSE: {greeting}")
    print()
    
    # Check for title and name recognition
    has_title = 'المهندس' in greeting or 'مهندس' in greeting
    has_name = 'أحمد' in greeting or 'احمد' in greeting
    has_both = has_title and has_name
    
    print("🔍 ANALYSIS:")
    print(f"   Title Recognition (المهندس): {'✅ YES' if has_title else '❌ NO'}")
    print(f"   Name Recognition (أحمد): {'✅ YES' if has_name else '❌ NO'}")
    print(f"   Combined Recognition: {'✅ SUCCESS' if has_both else '❌ PARTIAL'}")
    print(f"   No Questions: {'✅ YES' if '؟' not in greeting else '❌ NO'}")
    
    if has_both:
        print("\n🎉 PERFECT! Both title and name recognized!")
    else:
        print("\n⚠️  Improvement needed for complete title+name recognition")
    
    return has_both

if __name__ == "__main__":
    test_title_recognition()
