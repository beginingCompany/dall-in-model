#!/usr/bin/env python3
"""
Test the enhanced Arabic matching for variations like "مين مطورك".
"""

import os
from app.personality_analyzer import PersonalityAnalyzer

def test_arabic_variations():
    """Test Arabic variations that should now work."""
    print("🔍 Testing Enhanced Arabic Matching:")
    
    # Set dummy API key
    os.environ["OPENAI_API_KEY"] = "test_key"
    
    try:
        analyzer = PersonalityAnalyzer()
        
        # Test cases that should now work
        test_cases = [
            "مين مطورك",      # "Who is your developer" (your original issue)
            "مين طورك",       # "Who developed you" (shorter version)
            "منو مطورك",      # Different dialect of "who"
            "مين صنعك",       # "Who made you"
            "ايش هدفك",       # "What is your purpose" (different dialect)
            "شو دورك",        # "What is your role" (different dialect)
            "وش وظيفتك",      # "What is your job" (Gulf dialect)
            "مين أنت",        # "Who are you" (different dialect)
            "منو انت",        # "Who are you" (different dialect)
        ]
        
        print("  Testing Arabic variations:")
        for test_text in test_cases:
            is_identity, category, response_data = analyzer._fallback_identity_detection(test_text)
            status = "✅ DETECTED" if is_identity else "❌ MISSED"
            category_info = f"({category})" if category else ""
            print(f"    '{test_text}' → {status} {category_info}")
        
        print("\n  Summary:")
        detected_count = 0
        for test_text in test_cases:
            is_identity, _, _ = analyzer._fallback_identity_detection(test_text)
            if is_identity:
                detected_count += 1
        
        print(f"    Detected: {detected_count}/{len(test_cases)} variations")
        print(f"    Success rate: {(detected_count/len(test_cases)*100):.1f}%")
        
        if detected_count >= 7:  # Should detect at least 7/9
            print("    🎉 EXCELLENT! Enhanced matching is working!")
        elif detected_count >= 5:
            print("    👍 GOOD! Most variations detected.")
        else:
            print("    ⚠️ NEEDS IMPROVEMENT! Some variations still missed.")
            
    except Exception as e:
        print(f"  ⚠️ Error during testing: {e}")

def test_original_keywords():
    """Test that original keywords still work."""
    print("\n🔍 Testing Original Keywords Still Work:")
    
    os.environ["OPENAI_API_KEY"] = "test_key"
    
    try:
        analyzer = PersonalityAnalyzer()
        
        original_cases = [
            "من مطورك",       # Original Arabic
            "who is your developer",  # Original English
            "what is your purpose",   # Original English
            "ما هو هدفك",     # Original Arabic
        ]
        
        for test_text in original_cases:
            is_identity, category, response_data = analyzer._fallback_identity_detection(test_text)
            status = "✅ STILL WORKS" if is_identity else "❌ BROKEN"
            category_info = f"({category})" if category else ""
            print(f"    '{test_text}' → {status} {category_info}")
            
    except Exception as e:
        print(f"  ⚠️ Error during testing: {e}")

def main():
    print("🧪 Testing Enhanced Arabic Matching\n")
    test_arabic_variations()
    test_original_keywords()
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()
