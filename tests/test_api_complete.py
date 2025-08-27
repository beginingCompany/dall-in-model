#!/usr/bin/env python3
"""
Simple test script for the completed API
"""

import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working: {data['message']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_personality_analysis():
    """Test the personality analysis endpoint"""
    print("Testing personality analysis...")
    
    test_cases = [
        {
            "id": 1,
            "user_input": "I am a happy person who enjoys working with teams and solving complex problems",
            "new_input": [],
            "languages": "en"
        },
        {
            "id": 2,
            "user_input": "what color is the sky",  # Off-topic test
            "new_input": [],
            "languages": "en"
        },
        {
            "id": 3,
            "user_input": "who are you",  # Identity test
            "new_input": [],
            "languages": "en"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/analyze-personality",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                print(f"✅ Test {i} passed - Status: {status}")
                
                # Show response summary
                if status == "off_topic":
                    print(f"   Off-topic response: {data.get('description_off_topic', '')[:50]}...")
                elif status == "identity":
                    print(f"   Identity response: {data.get('description_identity', '')[:50]}...")
                elif status in ["complete", "incomplete"]:
                    print(f"   English desc: {data.get('description_english', '')[:50]}...")
                    print(f"   Missing traits: {data.get('missing_traits', [])}")
                
            else:
                print(f"❌ Test {i} failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Test {i} error: {e}")

def test_predict_endpoint():
    """Test the prediction endpoint"""
    print("Testing prediction endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": "I am a creative and analytical person who enjoys teamwork"},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction endpoint working")
            print(f"   Predictions: {len(data.get('predictions', []))} items")
            return True
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing BEGINING Personality Analysis API")
    print("=" * 50)
    
    # Test endpoints
    tests = [
        test_health_check,
        test_root_endpoint,
        test_personality_analysis,
        test_predict_endpoint
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API server and logs.")

if __name__ == "__main__":
    main()
