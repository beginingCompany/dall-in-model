#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_arabic_scenarios():
    """Test different scenarios that might trigger English fallback questions"""
    
    analyzer = PersonalityAnalyzer()
    
    print("=== Testing Arabic Language Support - Final Check ===")
    
    # Test 1: Normal Arabic input
    print("\n1. Normal Arabic input:")
    result1 = analyzer.analyze(
        id=1,
        user_input="أنا شخص اجتماعي أحب مقابلة الناس",
        languages="ar"
    )
    print("Input:", "أنا شخص اجتماعي أحب مقابلة الناس")
    print("Clarification questions:", result1.get("clarification_questions", []))
    for q in result1.get("clarification_questions", []):
        print(f"Question: '{q}' - Contains Arabic: {any('ا' <= char <= 'ي' for char in q)}")
    
    # Test 2: Empty input (triggers fallback)
    print("\n2. Empty input (triggers fallback):")
    result2 = analyzer.analyze(
        id=2,
        user_input="",
        languages="ar"
    )
    print("Input: (empty)")
    print("Clarification questions:", result2.get("clarification_questions", []))
    for q in result2.get("clarification_questions", []):
        print(f"Question: '{q}' - Contains Arabic: {any('ا' <= char <= 'ي' for char in q)}")
    
    # Test 3: Very short input
    print("\n3. Very short input:")
    result3 = analyzer.analyze(
        id=3,
        user_input="نعم",
        languages="ar"
    )
    print("Input:", "نعم")
    print("Clarification questions:", result3.get("clarification_questions", []))
    for q in result3.get("clarification_questions", []):
        print(f"Question: '{q}' - Contains Arabic: {any('ا' <= char <= 'ي' for char in q)}")
    
    # Test 4: Identity question in Arabic
    print("\n4. Identity question in Arabic:")
    result4 = analyzer.analyze(
        id=4,
        user_input="من أنت؟",
        languages="ar"
    )
    print("Input:", "من أنت؟")
    print("Identity response:", result4.get("description_identity", ""))
    print("Clarification questions:", result4.get("clarification_questions", []))
    for q in result4.get("clarification_questions", []):
        print(f"Question: '{q}' - Contains Arabic: {any('ا' <= char <= 'ي' for char in q)}")

if __name__ == "__main__":
    test_arabic_scenarios()
