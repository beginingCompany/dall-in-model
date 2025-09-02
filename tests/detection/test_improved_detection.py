#!/usr/bin/env python3
"""
Test the improved keyword detection to ensure it doesn't trigger false positives
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_improved_detection():
    """Test that the detection is more specific and context-aware"""
    analyzer = PersonalityAnalyzer()
    
    # Test cases that should NOT trigger identity detection (user self-descriptions)
    self_description_tests = [
        "I am a developer who enjoys creating applications",
        "I work as a programmer in a tech company", 
        "My job is software development",
        "I'm a creative developer",
        "My role involves building websites",
        "My purpose in life is to help others through technology",
        "I develop mobile applications",
        "I create software solutions",
        "أنا مطور برمجيات",
        "وظيفتي في شركة تقنية",
        "عملي هو تطوير التطبيقات",
        "دوري في الفريق",
        "أطور مواقع الويب"
    ]
    
    # Test cases that SHOULD trigger identity detection (asking about the system)
    identity_question_tests = [
        "who are you",
        "what is your purpose",
        "tell me who created you",
        "who is your developer",
        "what do you do",
        "من أنت",
        "من مطورك",
        "ما هو دورك",
        "ما هدفك"
    ]
    
    # Test cases that should be off-topic
    off_topic_tests = [
        "what color is the sky",
        "how do I learn Python programming",
        "what's the weather today",
        "kjsdfklsjdf random text",
        "ما لون السماء",
        "كيف أتعلم البرمجة"
    ]
    
    print("🧪 Testing Improved Keyword Detection")
    print("="*50)
    
    print("\n📝 Testing Self-Descriptions (should be personality input, NOT identity):")
    for i, text in enumerate(self_description_tests, 1):
        result = analyzer.analyze(
            id=100+i,
            user_input=text,
            new_input=[],
            languages="auto"
        )
        response_data = json.loads(result["content"])
        status = response_data.get("status")
        
        if status == "identity":
            print(f"❌ Test {i}: '{text}' → WRONGLY detected as {status}")
        elif status in ["incomplete", "complete"]:
            print(f"✅ Test {i}: '{text}' → CORRECTLY detected as {status}")
        else:
            print(f"⚠️  Test {i}: '{text}' → Unexpected status: {status}")
    
    print("\n🤖 Testing Identity Questions (should trigger identity responses):")
    for i, text in enumerate(identity_question_tests, 1):
        result = analyzer.analyze(
            id=200+i,
            user_input=text,
            new_input=[],
            languages="auto"
        )
        response_data = json.loads(result["content"])
        status = response_data.get("status")
        
        if status == "identity":
            print(f"✅ Test {i}: '{text}' → CORRECTLY detected as {status}")
        else:
            print(f"❌ Test {i}: '{text}' → WRONGLY detected as {status}")
    
    print("\n🔄 Testing Off-Topic Questions (should trigger off-topic responses):")
    for i, text in enumerate(off_topic_tests, 1):
        result = analyzer.analyze(
            id=300+i,
            user_input=text,
            new_input=[],
            languages="auto"
        )
        response_data = json.loads(result["content"])
        status = response_data.get("status")
        
        if status == "off_topic":
            print(f"✅ Test {i}: '{text}' → CORRECTLY detected as {status}")
        else:
            print(f"❌ Test {i}: '{text}' → WRONGLY detected as {status}")

if __name__ == "__main__":
    test_improved_detection()
