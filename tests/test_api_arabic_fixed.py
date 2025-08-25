#!/usr/bin/env python3

import requests
import json

def test_original_failing_case():
    """Test the original failing case that should now work"""
    
    url = "http://127.0.0.1:8000/analyze-personality"
    
    # The original failing request
    test_request = {
        "id": 653876528763528675238678466547677578236,
        "user_input": "مين الي مطورك",
        "new_input": [],
        "languages": "ar"
    }
    
    print("🎯 TESTING ORIGINAL FAILING CASE VIA API")
    print("=" * 60)
    print(f"Request: {json.dumps(test_request, indent=2, ensure_ascii=False)}")
    print("\nSending request...")
    
    try:
        response = requests.post(url, json=test_request)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success! Status Code: {response.status_code}")
            
            identity = data.get("description_identity")
            if identity and identity != "null":
                print(f"\n🎉 IDENTITY RESPONSE FOUND!")
                print(f"Arabic Response: {identity}")
                print(f"\n📊 Full Response:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                print(f"\n❌ Identity Response Still Missing: {identity}")
                print(f"Full response: {json.dumps(data, indent=2, ensure_ascii=False)}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

    # Test a few more improved Arabic cases
    additional_tests = [
        {
            "id": 123457,
            "user_input": "مين فريقك",
            "languages": "ar",
            "description": "Who is your team"
        },
        {
            "id": 123458, 
            "user_input": "شو بتعمل",
            "languages": "ar",
            "description": "What do you do"
        },
        {
            "id": 123459,
            "user_input": "كيف تحلل الشخصية",
            "languages": "ar", 
            "description": "How do you analyze personality"
        }
    ]
    
    print(f"\n" + "=" * 60)
    print("TESTING ADDITIONAL IMPROVED ARABIC CASES")
    print("=" * 60)
    
    for test in additional_tests:
        print(f"\n📝 Testing: '{test['user_input']}' ({test['description']})")
        
        try:
            response = requests.post(url, json=test)
            if response.status_code == 200:
                data = response.json()
                identity = data.get("description_identity")
                
                if identity:
                    print(f"✅ Response: {identity[:60]}...")
                else:
                    print(f"❌ No identity response")
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_original_failing_case()
