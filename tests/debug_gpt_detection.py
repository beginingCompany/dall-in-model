#!/usr/bin/env python3
"""
Debug GPT detection with detailed logging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_gpt_detection():
    print("🔍 DEBUGGING GPT DETECTION WITH DETAILS")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test a simple English phrase first
    test_phrase = "who are you"
    print(f"🔍 Testing English phrase: '{test_phrase}'")
    
    try:
        is_identity, identity_type, response_data = analyzer.detect_identity_question(test_phrase)
        print(f"✅ English result: {is_identity}, {identity_type}")
        
    except Exception as e:
        print(f"💥 English error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Arabic phrase
    test_phrase = "من مطورك"
    print(f"\n🔍 Testing Arabic phrase: '{test_phrase}'")
    
    try:
        is_identity, identity_type, response_data = analyzer.detect_identity_question(test_phrase)
        print(f"✅ Arabic result: {is_identity}, {identity_type}")
        
    except Exception as e:
        print(f"💥 Arabic error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gpt_detection()
