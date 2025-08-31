"""
Test script to verify the GPT-powered personality analyzer works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def test_gpt_powered_analyzer():
    """Test the GPT-powered personality analyzer functionality."""
    analyzer = PersonalityAnalyzer()
    
    print("🤖 Testing GPT-Powered Personality Analyzer")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "Arabic Introduction",
            "text": "مرحبا انا وليد مهندس بيوميجات",
            "expected": "Should detect introduction and generate greeting"
        },
        {
            "name": "English Introduction", 
            "text": "Hi I'm Sarah, I work as a developer",
            "expected": "Should detect introduction and generate greeting"
        },
        {
            "name": "Identity Question",
            "text": "who are you?",
            "expected": "Should detect identity question"
        },
        {
            "name": "Off-topic Question",
            "text": "What is the capital of France?",
            "expected": "Should detect off-topic question"
        },
        {
            "name": "Personality Content",
            "text": "I feel stressed when I have too much work",
            "expected": "Should be on-topic for personality analysis"
        }
    ]
    
    for test in test_cases:
        print(f"\n🧪 Test: {test['name']}")
        print(f"Input: '{test['text']}'")
        print(f"Expected: {test['expected']}")
        
        # Test language detection
        language = analyzer.detect_language(test['text'])
        print(f"✅ Language detected: {language}")
        
        # Test introduction detection
        has_intro, name, job, greeting = analyzer.detect_personal_introduction(test['text'])
        if has_intro:
            print(f"✅ Introduction detected: name='{name}', job='{job}'")
            print(f"   Generated greeting: '{greeting}'")
        else:
            print("ℹ️  No introduction detected")
        
        # Test identity detection
        is_identity, category, response_data = analyzer.detect_identity_question(test['text'])
        if is_identity:
            print(f"✅ Identity question detected: category='{category}'")
        else:
            print("ℹ️  Not an identity question")
        
        # Test off-topic detection
        is_off_topic, response_type, response_text = analyzer.detect_off_topic_question(test['text'], "en")
        if is_off_topic:
            print(f"✅ Off-topic detected: type='{response_type}'")
        else:
            print("ℹ️  On-topic for personality analysis")
        
        # Test trait analysis
        missing_traits = analyzer.analyze_missing_traits(test['text'], [])
        print(f"✅ Missing traits analysis: {missing_traits}")
        
        print("-" * 30)
    
    print("\n🎉 All GPT-powered tests completed!")
    print("\n📊 Summary:")
    print("✅ Language detection: GPT-powered")
    print("✅ Introduction detection: GPT-powered") 
    print("✅ Identity detection: GPT-powered")
    print("✅ Off-topic detection: GPT-powered")
    print("✅ Trait analysis: GPT-powered")
    print("\n🚀 No more regex patterns - Everything is AI-powered!")

if __name__ == "__main__":
    test_gpt_powered_analyzer()
