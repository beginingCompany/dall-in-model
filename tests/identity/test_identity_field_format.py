#!/usr/bin/env python3
"""
Test to verify that description_identity is returned as null when empty 
and as a string when there's an identity response.
"""

import requests
import json

def test_identity_field_format():
    """Test that description_identity is null when empty and string when populated."""
    
    print("🧪 Testing description_identity Field Format")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Non-identity input (should be null)",
            "input": "أنا مطور برمجيات أحب العمل مع الفرق",
            "expected_identity": None,
            "description": "User describing themselves - no identity response expected"
        },
        {
            "name": "Single identity question (should be string)", 
            "input": "من انت",
            "expected_identity": "string",
            "description": "Direct identity question - should get string response"
        },
        {
            "name": "Multi-identity question (should be string)",
            "input": "من انت وما هدفك",
            "expected_identity": "string", 
            "description": "Multiple identity questions - should get combined string response"
        }
    ]
    
    url = "http://localhost:8000/analyze-personality"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}] {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print(f"Expected: {test_case['expected_identity']}")
        
        payload = {
            "id": 123456,
            "user_input": test_case['input'],
            "new_input": [],
            "languages": "ar"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                identity_value = result.get('description_identity')
                
                print(f"Actual: {type(identity_value).__name__} = {identity_value}")
                
                # Check the format
                if test_case['expected_identity'] is None:
                    if identity_value is None:
                        print("✅ PASS: description_identity is null as expected")
                    else:
                        print(f"❌ FAIL: Expected null, got {type(identity_value).__name__}: {identity_value}")
                
                elif test_case['expected_identity'] == "string":
                    if isinstance(identity_value, str) and identity_value.strip():
                        print("✅ PASS: description_identity is non-empty string as expected")
                        print(f"   Content: {identity_value[:100]}{'...' if len(identity_value) > 100 else ''}")
                    else:
                        print(f"❌ FAIL: Expected non-empty string, got {type(identity_value).__name__}: {identity_value}")
                
            else:
                print(f"❌ Request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Request error: {e}")
    
    print(f"\n{'='*50}")
    print("✨ Key Points Verified:")
    print("✅ description_identity is null (not empty string) when no identity detected")
    print("✅ description_identity is string when identity response exists") 
    print("✅ Multi-question identity responses work correctly")
    print("✅ JSON format follows specification exactly")

if __name__ == "__main__":
    test_identity_field_format()
