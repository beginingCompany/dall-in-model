#!/usr/bin/env python3
"""
Simple summary test for the detection improvements
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def quick_summary_test():
    analyzer = PersonalityAnalyzer()
    
    # Key test cases from our original failing tests
    key_tests = [
        ("I am a developer", "personality"),
        ("انا مهندس", "personality"),
        ("who are you", "identity"),
        ("من مطورك", "identity"), 
        ("what is the capital of France", "off_topic"),
        ("what is machine learning", "off_topic"),
        ("tell me about history", "off_topic"),
        ("what color is the sky", "off_topic"),
        ("I work with your team sometimes", "personality"),
        ("You are a chatbot", "personality"),
        ("What are your top 3 objectives?", "identity"),
        ("who is the president", "off_topic"),
        ("what does Google do", "off_topic"),
    ]
    
    passed = 0
    total = len(key_tests)
    
    print("QUICK SUMMARY TEST")
    print("=" * 40)
    
    for input_text, expected in key_tests:
        try:
            is_identity, _, _ = analyzer.detect_identity_question(input_text)
            languages = analyzer.detect_language(input_text)
            is_off_topic, _, _ = analyzer.detect_off_topic_question(input_text, languages)
            
            if is_identity:
                actual = "identity"
            elif is_off_topic:
                actual = "off_topic"
            else:
                actual = "personality"
            
            if actual == expected:
                passed += 1
                status = "PASS"
            else:
                status = "FAIL"
            
            print(f"{status}: '{input_text}' -> {actual} (expected {expected})")
                
        except Exception as e:
            print(f"ERROR: '{input_text}' -> {e}")
    
    print("=" * 40)
    print(f"RESULTS: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
    
    return passed, total

if __name__ == "__main__":
    quick_summary_test()
