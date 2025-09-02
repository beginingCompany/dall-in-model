#!/usr/bin/env python3
"""
Debug identity vs personal greeting conflict
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def debug_conflict():
    """Debug the conflict between identity and personal greeting detection"""
    
    print("🔍 DEBUGGING IDENTITY VS PERSONAL GREETING CONFLICT")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    test_text = "انا المهندس احمد"
    
    print(f"Testing text: '{test_text}'")
    print("-" * 30)
    
    # Test personal introduction detection
    print("1. Personal Introduction Detection:")
    has_intro, name, job_title, greeting = analyzer.detect_personal_introduction(test_text)
    print(f"   Result: has_intro={has_intro}, name='{name}', job='{job_title}'")
    print(f"   Greeting: '{greeting}'")
    
    print()
    
    # Test identity detection
    print("2. Identity Question Detection:")
    is_identity, response_key, response_data = analyzer.detect_identity_question(test_text)
    print(f"   Result: is_identity={is_identity}, response_key='{response_key}'")
    print(f"   Response: '{response_data}'")
    
    print()
    
    if has_intro and is_identity:
        print("❌ CONFLICT: Text is detected as BOTH personal introduction AND identity question!")
        print("   This means personal greeting will be lost because identity takes precedence.")
    elif has_intro and not is_identity:
        print("✅ GOOD: Text is personal introduction, not identity question")
    elif not has_intro and is_identity:
        print("✅ GOOD: Text is identity question, not personal introduction")
    else:
        print("⚠️  NEITHER: Text is neither personal introduction nor identity question")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    debug_conflict()
