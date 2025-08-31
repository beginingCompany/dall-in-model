#!/usr/bin/env python3
"""
Test script for off-topic question detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_off_topic_detection():
    """Test the off-topic detection functionality"""
    analyzer = PersonalityAnalyzer()
    
    # Test cases for off-topic questions
    test_cases = [
        {
            "input": "what color is the sky",
            "languages": "en",
            "expected_status": "off_topic",
            "description": "Weather/factual question in English"
        },
        {
            "input": "ما لون السماء",
            "languages": "ar", 
            "expected_status": "off_topic",
            "description": "Weather question in Arabic"
        },
        {
            "input": "kjsklbgvjklsfdbvkjsbvk bksjbjklzs",
            "languages": "en",
            "expected_status": "off_topic",
            "description": "Gibberish text"
        },
        {
            "input": "how do I program in Python",
            "languages": "en",
            "expected_status": "off_topic", 
            "description": "Technical programming question"
        },
        {
            "input": "I am a happy person who likes to work with others",
            "languages": "en",
            "expected_status": "incomplete",  # This should NOT be off-topic
            "description": "Personality-related input"
        },
        {
            "input": "who are you",
            "languages": "en",
            "expected_status": "identity", # This should be identity, not off-topic
            "description": "Identity question"
        }
    ]
    
    print("Testing Off-Topic Detection\n" + "="*50)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            # Test the analyze method
            result = analyzer.analyze(
                id=i,
                user_input=test_case["input"],
                new_input=[],
                languages=test_case["languages"]
            )
            
            # Parse the response
            response_data = json.loads(result["content"])
            actual_status = response_data.get("status")
            
            # Check if the result matches expectations
            success = actual_status == test_case["expected_status"]
            status_symbol = "✅" if success else "❌"
            
            print(f"{status_symbol} Test {i}: {test_case['description']}")
            print(f"   Input: '{test_case['input']}'")
            print(f"   Expected: {test_case['expected_status']}")
            print(f"   Actual: {actual_status}")
            
            # Show off-topic response if applicable
            if actual_status == "off_topic":
                off_topic_response = response_data.get("description_off_topic", "")
                print(f"   Off-topic response: {off_topic_response[:100]}...")
            elif actual_status == "identity":
                identity_response = response_data.get("description_identity", "")
                print(f"   Identity response: {identity_response[:100]}...")
            
            print()
            
        except Exception as e:
            print(f"❌ Test {i}: ERROR - {str(e)}")
            print(f"   Input: '{test_case['input']}'")
            print()

if __name__ == "__main__":
    test_off_topic_detection()
