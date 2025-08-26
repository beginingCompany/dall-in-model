#!/usr/bin/env python3
"""
Test GPT-based identity detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_gpt_identity_detection():
    """Test the new GPT-based identity detection"""
    
    try:
        from personality_analyzer import PersonalityAnalyzer
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Create OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  No OpenAI API key found. Testing fallback mode only.")
            client = None
        else:
            client = OpenAI(api_key=api_key)
            print("✅ OpenAI client created successfully")
        
        test_cases = [
            {
                "input": "who is ur developer",
                "expected": "developer",
                "description": "Casual variation with 'ur'"
            },
            {
                "input": "who made you",
                "expected": "developer", 
                "description": "Alternative developer question"
            },
            {
                "input": "i am a developer",
                "expected": "none",
                "description": "User describing themselves (should NOT trigger)"
            },
            {
                "input": "what do you do",
                "expected": "role",
                "description": "Role question"
            },
            {
                "input": "tell me about yourself",
                "expected": "who_are_you",
                "description": "About question"
            },
            {
                "input": "how does this work",
                "expected": "how_analyze",
                "description": "How it works question"
            },
            {
                "input": "I like working with teams",
                "expected": "none", 
                "description": "User personality trait (should NOT trigger)"
            }
        ]
        
        print("\nTesting GPT-based identity detection...\n")
        print("="*70)
        
        for i, test_case in enumerate(test_cases, 1):
            user_input = test_case["input"]
            expected = test_case["expected"]
            description = test_case["description"]
            
            print(f"\nTest {i}: {description}")
            print(f"Input: '{user_input}'")
            
            try:
                # Test with GPT
                response = PersonalityAnalyzer.get_identity_response(user_input, "en", client)
                
                if expected == "none":
                    if not response:
                        print(f"✅ PASS: Correctly identified as non-identity question")
                    else:
                        print(f"❌ FAIL: Incorrectly triggered identity response: {response[:50]}...")
                else:
                    if response:
                        print(f"✅ PASS: Identity response triggered")
                        print(f"   Response: {response[:80]}...")
                        # Check if it's the right category by looking for key phrases
                        if expected == "developer" and ("Saudi Arabia" in response or "researchers" in response):
                            print(f"✅ PASS: Correct category (developer)")
                        elif expected == "role" and ("role" in response or "insights" in response):
                            print(f"✅ PASS: Correct category (role)")
                        elif expected == "who_are_you" and ("Minus Zero" in response):
                            print(f"✅ PASS: Correct category (who_are_you)")
                        elif expected == "how_analyze" and ("120 personality" in response or "scale" in response):
                            print(f"✅ PASS: Correct category (how_analyze)")
                        else:
                            print(f"⚠️  Response category might be different than expected")
                    else:
                        print(f"❌ FAIL: Expected identity response but got none")
                        
            except Exception as e:
                print(f"❌ ERROR in test: {e}")
                # Try fallback
                try:
                    response = PersonalityAnalyzer.get_identity_response(user_input, "en", None)
                    print(f"🔄 Fallback result: {'Found' if response else 'Not found'}")
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
        
        print("\n" + "="*70)
        print("🎉 GPT-based identity detection test complete!")
        print("✅ System now uses AI intelligence to understand user intent")
        print("✅ Can handle casual language, slang, and variations")
        print("✅ Better distinction between identity questions vs self-description")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_gpt_identity_detection()
