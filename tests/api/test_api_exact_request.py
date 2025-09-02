#!/usr/bin/env python3
"""
Test the API with the exact same request to compare results
"""

import requests
import json
import time

def test_api_with_exact_request():
    print("Testing API with exact request...")
    
    # Wait for server to be ready
    time.sleep(3)
    
    url = "http://localhost:8000/analyze"
    
    # Exact request data from the user
    data = {
        "id": 225985882206,
        "user_input": "مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى.",
        "new_input": [
            {
                "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
                "answer": "من انت"
            }
        ],
        "languages": "ar"
    }
    
    print(f"Request URL: {url}")
    print(f"Request data: {json.dumps(data, ensure_ascii=False, indent=2)}")
    print("=" * 80)
    
    try:
        response = requests.post(url, json=data, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            missing_traits = result.get('missing_traits', [])
            print(f"\n=== COMPARISON ===")
            print(f"User reported missing traits: ['emotional', 'social', 'cognitive', 'behavioral'] (4 traits)")
            print(f"API returned missing traits: {missing_traits} ({len(missing_traits)} traits)")
            
            if len(missing_traits) == 4:
                print("❌ ISSUE: API still returning all 4 traits as missing")
                print("This suggests the API server may not have reloaded with the fixes")
            elif len(missing_traits) == 1 and missing_traits == ['social']:
                print("✅ SUCCESS: API now working correctly - only social trait missing as expected")
            else:
                print(f"⚠️ DIFFERENT: API returned {len(missing_traits)} missing traits")
            
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        print("Make sure the API server is running on http://localhost:8000")

if __name__ == "__main__":
    test_api_with_exact_request()
