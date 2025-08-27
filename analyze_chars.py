#!/usr/bin/env python3
"""
Analyze the Arabic character differences.
"""

def analyze_characters():
    print("🔍 Analyzing Arabic Character Differences:")
    
    chars = [1575, 1571, 1620]
    for char_code in chars:
        char = chr(char_code)
        print(f"  Character {char_code}: '{char}' - {unicodedata.name(char, 'UNKNOWN')}")
    
    print("\n  The issue:")
    print("  - 1575 = ARABIC LETTER ALEF")
    print("  - 1571 = ARABIC LETTER ALEF WITH HAMZA ABOVE")  
    print("  - 1620 = ARABIC DIACRITICAL MARK")
    print("  These look similar but are different Unicode characters!")

if __name__ == "__main__":
    import unicodedata
    analyze_characters()
