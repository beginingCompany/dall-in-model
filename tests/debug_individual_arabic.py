#!/usr/bin/env python3
"""
Debug individual Arabic identity detection
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_individual_arabic():
    print("🔍 DEBUGGING INDIVIDUAL ARABIC DETECTION")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test the exact phrase that was failing
    test_phrase = "من مطورك"
    print(f"🔍 Testing phrase: '{test_phrase}'")
    
    try:
        is_identity, identity_type, response_data = analyzer.detect_identity_question(test_phrase)
        print(f"✅ Detection result: {is_identity}, {identity_type}")
        
        if is_identity:
            print(f"🎯 Identity type detected: {identity_type}")
            if response_data:
                print(f"📝 Response data available: {list(response_data.keys())}")
        else:
            print("❌ No identity detected")
            
    except Exception as e:
        print(f"💥 Error during detection: {e}")
        import traceback
        traceback.print_exc()
        
    # Also test a few other Arabic phrases
    test_phrases = [
        "من أنت",
        "ما هو مشروع بيجينينغ",
        "ما هو دورك",
        "من مطورك؟",  # with question mark
        "من هو مطورك",  # alternative phrasing
    ]
    
    print("\n🔍 Testing other Arabic phrases:")
    for phrase in test_phrases:
        try:
            is_identity, identity_type, response_data = analyzer.detect_identity_question(phrase)
            print(f"  '{phrase}' → {is_identity}, {identity_type}")
        except Exception as e:
            print(f"  '{phrase}' → ERROR: {e}")

if __name__ == "__main__":
    debug_individual_arabic()
