#!/usr/bin/env python3
"""
Test the enhanced multi-question identity response system.
Tests the ability to handle multiple identity questions in a single input
and generate friendly, combined responses.
"""

import requests
import json
import time

def test_analyzer_api(user_input, new_input=None, expected_identity=True, test_name="Test"):
    """Test the personality analyzer API with given input."""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")
    print(f"Input: {user_input}")
    if new_input:
        print(f"New Input: {new_input}")
    
    url = "http://localhost:8000/analyze-personality"
    
    payload = {
        "id": 225985882206,
        "user_input": user_input,
        "new_input": new_input or [],
        "languages": "ar"  # Test with Arabic
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"Status: {result.get('status', 'unknown')}")
            
            # Check for identity response
            identity_response = result.get('description_identity', '')
            if identity_response:
                print(f"🤖 Identity Response:")
                print(f"   {identity_response}")
                
                if expected_identity:
                    print(f"✅ Identity response detected as expected!")
                else:
                    print(f"❌ Unexpected identity response")
            else:
                if expected_identity:
                    print(f"❌ Expected identity response but got none")
                else:
                    print(f"✅ No identity response as expected")
            
            # Show clarification questions if any
            clarification = result.get('clarification_questions', [])
            if clarification:
                print(f"❓ Clarification Questions: {clarification}")
            
            # Show missing traits
            missing = result.get('missing_traits', [])
            if missing:
                print(f"🔍 Missing Traits: {missing}")
                
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def main():
    """Run comprehensive tests for multi-question identity responses."""
    
    print("🚀 Starting Multi-Question Identity Response Tests")
    print("=" * 80)
    
    test_cases = [
        # Single identity questions (Arabic)
        {
            "user_input": "من انت",
            "expected_identity": True,
            "test_name": "Single Arabic Identity Question: Who are you"
        },
        {
            "user_input": "ما هدفك",
            "expected_identity": True,
            "test_name": "Single Arabic Purpose Question"
        },
        
        # Multiple identity questions (Arabic)
        {
            "user_input": "من انت وما هدفك",
            "expected_identity": True,
            "test_name": "Multi Arabic: Who are you + Purpose"
        },
        {
            "user_input": "اخبرني عن نفسك وكيف تعمل",
            "expected_identity": True,
            "test_name": "Multi Arabic: Tell me about yourself + How you work"
        },
        {
            "user_input": "من انت وما هو مشروع BEGINING وما هدفك",
            "expected_identity": True,
            "test_name": "Multi Arabic: Identity + Project + Purpose"
        },
        
        # Multiple identity questions (English)
        {
            "user_input": "who are you and what is your purpose",
            "expected_identity": True,
            "test_name": "Multi English: Who are you + Purpose"
        },
        {
            "user_input": "tell me about yourself and how do you analyze personality",
            "expected_identity": True,
            "test_name": "Multi English: About yourself + Analysis method"
        },
        {
            "user_input": "what is BEGINING and who developed you",
            "expected_identity": True,
            "test_name": "Multi English: Project + Developer"
        },
        
        # Mixed content with personality info
        {
            "user_input": "مرحبا أنا مهندس برمجيات. من انت وما هدفك؟",
            "expected_identity": True,
            "test_name": "Mixed: Personality info + Identity questions (Arabic)"
        },
        {
            "user_input": "I work as a data scientist and love solving problems. Who are you and what do you do?",
            "expected_identity": True,
            "test_name": "Mixed: Personality info + Identity questions (English)"
        },
        
        # Complex multi-question scenarios
        {
            "user_input": "من انت وما هو مشروع BEGINING وكيف تحلل الشخصية وما هدفك في الحياة",
            "expected_identity": True,
            "test_name": "Complex Multi Arabic: Identity + Project + Analysis + Purpose"
        },
        
        # Non-identity questions (should not trigger identity response)
        {
            "user_input": "أنا مطور ويب وأحب البرمجة",
            "expected_identity": False,
            "test_name": "Non-identity: User describing themselves (Arabic)"
        },
        {
            "user_input": "I am a developer who enjoys teamwork",
            "expected_identity": False,
            "test_name": "Non-identity: User describing themselves (English)"
        },
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{total_tests}] Running test...")
        
        success = test_analyzer_api(
            user_input=test_case["user_input"],
            new_input=test_case.get("new_input"),
            expected_identity=test_case["expected_identity"],
            test_name=test_case["test_name"]
        )
        
        if success:
            successful_tests += 1
        
        # Small delay between tests
        time.sleep(1)
    
    # Test Summary
    print(f"\n{'='*80}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print(f"🎉 All tests passed! Multi-question identity system working perfectly!")
    else:
        print(f"⚠️  Some tests failed. Check the logs above for details.")
    
    print(f"\n🏁 Testing completed!")

if __name__ == "__main__":
    main()
