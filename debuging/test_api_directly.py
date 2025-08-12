"""
API Test Script - Directly test the analyzer API with a simple example
"""
import os
import sys
import json
import requests
from pprint import pprint

# Add the project root to the path so we can import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_api():
    """Test the analyzer API with a simple example"""
    print("\n=== Testing Analyzer API ===")
    
    # Base URL for the API - change if needed
    base_url = "http://127.0.0.1:8000"
    
    # Test data
    test_data = {
        "id": 123,
        "user_input": "I am a software engineer who likes to solve complex problems. I enjoy working in teams and helping others.",
        "new_input": [
            {
                "question": "How do you handle stress at work?",
                "answer": "I take regular breaks and prioritize tasks to manage stress effectively."
            },
            {
                "question": "What are your hobbies outside of work?",
                "answer": "I enjoy hiking and reading science fiction books."
            }
        ],
        "languages": ["en"]
    }
    
    # Print the data being sent
    print("\nSending data to API:")
    pprint(test_data)
    
    try:
        # Make the API request
        print("\nMaking API request...")
        response = requests.post(
            f"{base_url}/analyze-personality", 
            json=test_data,
            timeout=60  # Increase timeout for slower systems
        )
        
        # Print the status code
        print(f"\nStatus Code: {response.status_code}")
        
        # Print the raw response text
        print("\nRaw Response Text:")
        print(response.text[:1000])  # Limit to first 1000 chars in case it's very large
        
        # Try to parse as JSON
        try:
            if response.text.strip():
                result = response.json()
                print("\nParsed JSON Response:")
                pprint(result)
            else:
                print("\nEmpty response received from server")
        except json.JSONDecodeError as e:
            print(f"\nFailed to parse JSON response: {e}")
    
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the API server. Make sure the server is running.")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    print("API Test Script")
    test_api()
