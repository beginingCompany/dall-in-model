#!/usr/bin/env python3
"""
Debug specific trait detection for the first case
"""

import re
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_trait_detection():
    """Debug trait detection for the specific case"""
    
    print("🔍 DEBUGGING TRAIT DETECTION")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # First case data
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        }
        # Excluding the identity question
    ]
    
    # Combine all text as the method does
    all_text = user_input.lower()
    for qa in new_input:
        answer = qa.get("answer", "").lower()
        # Skip identity questions in trait analysis
        is_identity, _, _ = analyzer.detect_identity_question(answer)
        if not is_identity:
            all_text += " " + answer
    
    print(f"Combined text: {all_text}")
    print()
    
    # Check each trait pattern
    for trait, pattern in analyzer.TRAIT_PATTERNS.items():
        matches = re.findall(pattern, all_text)
        is_present = bool(re.search(pattern, all_text))
        print(f"{trait.upper()}:")
        print(f"  Present: {is_present}")
        if matches:
            print(f"  Matches: {matches}")
        else:
            print(f"  No matches found")
        print()
    
    # Check what the analyze_missing_traits method returns
    missing_traits = analyzer.analyze_missing_traits(user_input, new_input)
    print(f"Missing traits: {missing_traits}")
    
    # Let's specifically look for common words that should match
    print("\nManual word check:")
    test_words = ["analytical", "solving", "satisfaction", "working", "teams", "leadership", "mentoring"]
    for word in test_words:
        if word in all_text:
            print(f"  Found: {word}")

if __name__ == "__main__":
    debug_trait_detection()
