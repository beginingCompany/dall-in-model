#!/usr/bin/env python3
"""
Test script to verify the PersonalityAnalyzer improvements.
"""

from app.personality_analyzer import PersonalityAnalyzer

def test_language_detection():
    """Test automatic language detection."""
    print("🔍 Testing Language Detection:")
    
    # Test English
    english_text = "Hello, how are you today?"
    detected = PersonalityAnalyzer.detect_language(english_text)
    print(f"  English text: '{english_text}' → {detected}")
    
    # Test Arabic
    arabic_text = "مرحبا، كيف حالك اليوم؟"
    detected = PersonalityAnalyzer.detect_language(arabic_text)
    print(f"  Arabic text: '{arabic_text}' → {detected}")
    
    # Test mixed
    mixed_text = "Hello مرحبا"
    detected = PersonalityAnalyzer.detect_language(mixed_text)
    print(f"  Mixed text: '{mixed_text}' → {detected}")
    print()

def test_question_generation():
    """Test clarification question generation with repetition prevention."""
    print("🔍 Testing Question Generation:")
    
    missing_traits = ["emotional", "social"]
    languages = "en"
    
    # First call
    questions1 = PersonalityAnalyzer.generate_clarification_questions(
        missing_traits, languages, max_questions=2
    )
    print(f"  First call: {len(questions1)} questions")
    for i, q in enumerate(questions1, 1):
        print(f"    {i}. {q}")
    
    # Second call with asked questions
    questions2 = PersonalityAnalyzer.generate_clarification_questions(
        missing_traits, languages, max_questions=2, asked_questions=questions1
    )
    print(f"  Second call (avoiding repetition): {len(questions2)} questions")
    for i, q in enumerate(questions2, 1):
        print(f"    {i}. {q}")
    
    # Check if questions are different
    overlap = set(questions1) & set(questions2)
    print(f"  Repeated questions: {len(overlap)}")
    print()

def test_arabic_normalization():
    """Test Arabic text normalization."""
    print("🔍 Testing Arabic Normalization:")
    
    # Create analyzer instance
    import os
    os.environ["OPENAI_API_KEY"] = "test_key"  # Dummy key for testing
    
    try:
        analyzer = PersonalityAnalyzer()
        
        # Test with diacritics
        text_with_diacritics = "مطورُك"
        text_without_diacritics = "مطورك"
        
        print(f"  Text with diacritics: '{text_with_diacritics}'")
        print(f"  Text without diacritics: '{text_without_diacritics}'")
        
        # The normalization should make them match in fallback detection
        result1 = analyzer._fallback_identity_detection(text_with_diacritics)
        result2 = analyzer._fallback_identity_detection(text_without_diacritics)
        
        print(f"  Detection result 1: {result1[0]} ({result1[1]})")
        print(f"  Detection result 2: {result2[0]} ({result2[1]})")
        print(f"  Both detected: {result1[0] and result2[0]}")
        
    except Exception as e:
        print(f"  ⚠️ Could not test normalization (need real API key): {e}")
    print()

def main():
    """Run all tests."""
    print("🧪 Testing PersonalityAnalyzer Improvements\n")
    
    test_language_detection()
    test_question_generation()
    test_arabic_normalization()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()
