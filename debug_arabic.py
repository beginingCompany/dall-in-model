#!/usr/bin/env python3
"""
Debug why "مين أنت" is not being detected.
"""

import os
import unicodedata
from app.personality_analyzer import PersonalityAnalyzer

def debug_text_processing():
    """Debug how the text is being processed."""
    print("🔍 Debugging Text Processing:")
    
    test_text = "مين أنت"
    print(f"  Original text: '{test_text}'")
    
    # Show each processing step
    text_lower = test_text.lower().strip()
    print(f"  After lower(): '{text_lower}'")
    
    text_normalized = unicodedata.normalize("NFKD", text_lower)
    print(f"  After normalize(): '{text_normalized}'")
    
    # Check if it matches known keywords
    who_are_you_keywords = [
        "who are you", "who r u", "who ru", "who u", "tell me about you", "introduce yourself",
        "about you", "who is this", "ur identity", "your identity",
        "من أنت", "مين أنت", "منو أنت", "من انت", "مين انت", "منو انت",
        "عرف بنفسك", "عرفني بنفسك", "قل لي من أنت", "قول لي من أنت",
        "هويتك", "هويك", "شخصيتك"
    ]
    
    print(f"\n  Checking against who_are_you keywords:")
    for keyword in who_are_you_keywords:
        if keyword in text_normalized:
            print(f"    ✅ MATCH: '{keyword}'")
        elif keyword == "مين أنت":
            print(f"    ❌ NO MATCH: '{keyword}' (this should match!)")
            # Let's see character by character
            print(f"      Text chars: {[ord(c) for c in text_normalized]}")
            print(f"      Keyword chars: {[ord(c) for c in keyword]}")

def test_direct_matching():
    """Test direct string matching."""
    print("\n🔍 Testing Direct String Matching:")
    
    test_text = "مين أنت"
    keyword = "مين أنت"
    
    print(f"  test_text == keyword: {test_text == keyword}")
    print(f"  keyword in test_text: {keyword in test_text}")
    print(f"  test_text in keyword: {test_text in keyword}")
    
    # Test with different character encodings
    print(f"\n  Character analysis:")
    print(f"    test_text: {repr(test_text)}")
    print(f"    keyword: {repr(keyword)}")

def main():
    print("🧪 Debugging Arabic Text Matching\n")
    debug_text_processing()
    test_direct_matching()
    print("\n✅ Debug completed!")

if __name__ == "__main__":
    main()
