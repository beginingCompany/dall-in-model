#!/usr/bin/env python3

import requests
import json

def test_fixed_api():
    """Test the fixed API with the exact same request that was failing"""
    
    url = "http://127.0.0.1:8000/analyze-personality"
    
    # The exact same request that was returning null
    test_request = {
        "id": 653876528763528675238678466547677578236,
        "user_input": "Who is your developer?",
        "new_input": [],
        "languages": "en"
    }
    
    print("=== Testing Fixed API ===")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    print("\nSending request...")
    
    try:
        response = requests.post(url, json=test_request)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success! Status Code: {response.status_code}")
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # Check specifically for description_identity
            identity = data.get("description_identity")
            if identity and identity != "null":
                print(f"\n🎯 Identity Response Found: {identity}")
            else:
                print(f"\n❌ Identity Response Still Missing: {identity}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

    # Test another identity question in Arabic
    print(f"\n" + "="*50)
    
    arabic_request = {
        "id": 123456,
        "user_input": "من أنت؟",
        "new_input": [],
        "languages": "ar"
    }
    
    print("Testing Arabic identity question...")
    print(f"Request: {json.dumps(arabic_request, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=arabic_request)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success! Status Code: {response.status_code}")
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            identity = data.get("description_identity")
            if identity and identity != "null":
                print(f"\n🎯 Arabic Identity Response Found: {identity}")
            else:
                print(f"\n❌ Arabic Identity Response Missing: {identity}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_fixed_api()
