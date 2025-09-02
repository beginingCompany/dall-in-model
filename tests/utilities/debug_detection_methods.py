#!/usr/bin/env python3
"""
Debug GPT API and fallback detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_detection_methods():
    print("🔍 DEBUGGING DETECTION METHODS")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test fallback detection directly
    print("Testing fallback detection:")
    
    test_phrases = [
        "who are you",
        "who is your developer", 
        "من مطورك",
        "developer",
        "made you"
    ]
    
    for phrase in test_phrases:
        try:
            is_identity, identity_type, response_data = analyzer._fallback_identity_detection(phrase)
            print(f"  Fallback '{phrase}' → {is_identity}, {identity_type}")
        except Exception as e:
            print(f"  Fallback '{phrase}' → ERROR: {e}")
    
    # Test if OpenAI client is working
    print("\n🔍 Testing OpenAI client:")
    try:
        if hasattr(analyzer, 'client') and analyzer.client:
            print("✅ OpenAI client exists")
            
            # Try a simple API call
            response = analyzer.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Reply with just the word 'test'"}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip()
            print(f"✅ OpenAI API test: '{result}'")
        else:
            print("❌ OpenAI client not found")
    except Exception as e:
        print(f"💥 OpenAI API error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_detection_methods()
