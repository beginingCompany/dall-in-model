"""
Debug the GPT response to see where the identity response is coming from
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer
import json

# Monkey patch to debug GPT calls
original_call_gpt = PersonalityAnalyzer.call_gpt

def debug_call_gpt(self, input_data):
    print(f"🔍 DEBUG: call_gpt called")
    print(f"   Input data: {json.dumps(input_data, indent=2)}")
    
    result = original_call_gpt(self, input_data)
    
    print(f"   GPT response content preview: {result.get('content', '')[:200]}...")
    print()
    
    return result

# Replace method
PersonalityAnalyzer.call_gpt = debug_call_gpt

def test_gpt_response():
    analyzer = PersonalityAnalyzer()
    
    print("🧪 Debug Test: GPT Response Analysis")
    print("=" * 50)
    
    result = analyzer.analyze(
        id=123,
        user_input="who are you",
        new_input=[
            {"question": "Tell me about your interests", "answer": "I like technology"},
            {"question": "How do you handle stress?", "answer": ""}
        ],
        languages="en"
    )
    
    print(f"Final result identity: {result.get('description_identity', 'None')}")

if __name__ == "__main__":
    test_gpt_response()
