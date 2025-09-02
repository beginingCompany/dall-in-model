#!/usr/bin/env python3
"""
Debug why some informal variations are not detected by GPT
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_gpt_vs_fallback():
    print("🔍 DEBUGGING GPT vs FALLBACK FOR INFORMAL VARIATIONS")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test some failing cases
    failing_cases = ["what ur purpose", "ur job", "who u"]
    
    for case in failing_cases:
        print(f"\n{'='*50}")
        print(f"🔍 Testing: '{case}'")
        
        try:
            # Test complete detection
            is_identity, detected_category, response_data = analyzer.detect_identity_question(case)
            print(f"📊 Complete detection: {is_identity}, {detected_category}")
            
            # Test fallback directly  
            is_fallback, fallback_category, fallback_data = analyzer._fallback_identity_detection(case)
            print(f"🔄 Fallback detection: {is_fallback}, {fallback_category}")
            
            # If complete detection failed but fallback worked, GPT is the issue
            if not is_identity and is_fallback:
                print(f"⚠️ GPT detection failed, but fallback works! GPT needs improvement.")
            elif is_identity and not is_fallback:
                print(f"✅ GPT detection worked, fallback not needed.")
            elif not is_identity and not is_fallback:
                print(f"❌ Both GPT and fallback failed.")
            else:
                print(f"✅ Both detections work.")
                
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    debug_gpt_vs_fallback()
