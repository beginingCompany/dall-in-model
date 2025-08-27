#!/usr/bin/env python3
"""
Targeted test for the critical failing cases from deep analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def test_critical_fixes():
    analyzer = PersonalityAnalyzer()
    
    # Test the critical failing cases
    critical_tests = [
        # Arabic self-description that was failing
        ("انا مهندس", "personality", "Arabic self-description"),
        
        # Off-topic questions that were being missed
        ("what is the capital of France", "off_topic", "Geography question"),
        ("what is machine learning", "off_topic", "Technical question"),
        ("explain artificial intelligence", "off_topic", "Technical explanation"),
        ("tell me about history", "off_topic", "General knowledge"),
        ("how does photosynthesis work", "off_topic", "Science question"),
        ("what is quantum physics", "off_topic", "Physics question"),
        ("أخبرني عن التاريخ", "off_topic", "Arabic history question"),
        
        # False positives that should be personality
        ("I work with your team sometimes", "personality", "False positive - team mention"),
        
        # Third-party questions that should be off-topic
        ("who is the president", "off_topic", "Third party question"),
        ("what does Google do", "off_topic", "Company question"),
        ("who created Facebook", "off_topic", "Company creator question"),
        
        # Short inputs
        ("you", "off_topic", "Very short input"),
        ("developer", "off_topic", "Single word"),
        
        # Mixed content
        ("I am a developer. What do you do?", "identity", "Mixed content with question"),
        
        # Statement vs question  
        ("You are a chatbot", "personality", "Statement about system"),
        
        # Edge cases
        ("What are your top 3 objectives?", "identity", "Identity with numbers"),
        ("Suppose your purpose was different", "identity", "Hypothetical about system"),
        ("Your purpose is clearer than mine", "identity", "Comparative about system"),
    ]
    
    passed = 0
    failed = 0
    
    print("🎯 TESTING CRITICAL FIXES")
    print("=" * 50)
    
    for input_text, expected, description in critical_tests:
        print(f"\n🧪 {description}")
        print(f"Input: '{input_text}'")
        print(f"Expected: {expected}")
        
        try:
            # Test detection
            is_identity, identity_category, _ = analyzer.detect_identity_question(input_text)
            languages = analyzer.detect_language(input_text)
            is_off_topic, off_topic_type, _ = analyzer.detect_off_topic_question(input_text, languages)
            
            # Determine actual category
            if is_identity:
                actual = "identity"
                detail = f"({identity_category})"
            elif is_off_topic:
                actual = "off_topic"
                detail = f"({off_topic_type})"
            else:
                actual = "personality"
                detail = "(incomplete/normal)"
            
            print(f"Actual: {actual} {detail}")
            
            if actual == expected:
                print("✅ FIXED!")
                passed += 1
            else:
                print("❌ Still failing")
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 CRITICAL FIXES RESULTS")
    print(f"Total: {len(critical_tests)}")
    print(f"Fixed: {passed}")
    print(f"Still failing: {failed}")
    print(f"Fix rate: {(passed/len(critical_tests))*100:.1f}%")
    
    return passed, failed

if __name__ == "__main__":
    test_critical_fixes()
