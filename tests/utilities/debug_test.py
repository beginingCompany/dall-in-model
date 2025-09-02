#!/usr/bin/env python3
from app.personality_analyzer import PersonalityAnalyzer

# Test the specific case that's failing
analyzer = PersonalityAnalyzer()

test_text = "I am a developer who enjoys creating applications"
print(f"Testing: '{test_text}'")

# Test each detection layer
print("\n1. Testing GPT Identity Classification:")
gpt_result = analyzer._gpt_identity_classification(test_text)
print(f"GPT result: {gpt_result}")

print("\n2. Testing Smart Similarity:")
smart_result = analyzer._is_similar_to_identity_keywords(test_text)
print(f"Smart similarity result: {smart_result}")

print("\n3. Testing Fallback Detection:")
fallback_result = analyzer._fallback_identity_detection(test_text)
print(f"Fallback result: {fallback_result}")

print("\n4. Testing Full Analysis:")
full_result = analyzer.analyze_personality(test_text)
print(f"Full analysis result: {full_result}")
