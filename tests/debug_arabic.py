#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import re

def debug_arabic_pattern():
    """Debug why 'مين الي مطورك' is not matching"""
    
    test_input = "مين الي مطورك"
    print(f"Testing: '{test_input}'")
    print(f"Length: {len(test_input)}")
    print(f"Characters: {[c for c in test_input]}")
    
    # Check if it has Arabic characters
    has_arabic = re.search(r'[\u0600-\u06FF]', test_input)
    print(f"Has Arabic characters: {bool(has_arabic)}")
    
    # Test direct method
    response = PersonalityAnalyzer.get_identity_response(test_input, "ar")
    print(f"Direct response: {response}")
    
    # Check against each developer pattern
    text = test_input.lower().strip()
    print(f"Normalized text: '{text}'")
    
    developer_patterns = [
        r"who\s+(made|built|created|developed)\s+you", 
        r"who\s+is\s+your\s+developer",
        r"your\s+(maker|creator|developer)", 
        r"who\s+designed\s+you", 
        r"who\s+programmed\s+you",
        r"من\s+(صنعك|بناك|طورك|صممك)", 
        r"من\s+هو\s+مطورك", 
        r"مين\s+عملك"
    ]
    
    print("\nTesting against enhanced patterns:")
    for i, pattern in enumerate(developer_patterns):
        match = re.search(pattern, text)
        print(f"  Pattern {i+1}: {pattern} -> {'✅' if match else '❌'}")
    
    # Test against original triggers
    original_triggers = ["who is your developer", "who made you", "who built you", "من هو مطورك", "من صنعك", "من بناك"]
    print(f"\nTesting against original triggers:")
    for trigger in original_triggers:
        if trigger.lower() in text:
            print(f"  '{trigger}' -> ✅ Found")
        else:
            print(f"  '{trigger}' -> ❌ Not found")
    
    # Test some variations
    variations = [
        "مين مطورك",
        "مين اللي مطورك", 
        "مين الي مطورك",
        "من مطورك",
        "من هو مطورك"
    ]
    
    print(f"\nTesting variations:")
    for var in variations:
        resp = PersonalityAnalyzer.get_identity_response(var, "ar")
        print(f"  '{var}' -> {'✅' if resp else '❌'}")

if __name__ == "__main__":
    debug_arabic_pattern()
