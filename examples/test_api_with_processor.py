"""
Test script demonstrating how to use the input processor in API calls.
"""

import json
import sys
import os
import requests

# Add the project root directory to the Python path
# This allows importing modules from the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.input_processor import format_for_analysis

def test_api_with_processor(base_url="http://127.0.0.1:8000"):
    """
    Test the personality analysis API with the input processor.
    
    Args:
        base_url (str): The base URL of your API
    """
    print("Testing personality analysis with input processor...")
    
    # Example personality data
    input_data = {
        "id": 123456,
        "user_input": "I am a programmer who enjoys solving complex problems. I work best when I have clear goals.",
        "new_input": [
            {
                "question": "How do you handle stress at work? Do you have any specific techniques?",
                "answer": "I take short breaks to clear my mind. Sometimes I go for a short walk."
            },
            {
                "question": "What motivates you in your career?",
                "answer": "I enjoy learning new technologies and challenging myself with difficult problems."
            }
        ],
        "languages": "en"
    }
    
    # Process the data with our input processor
    processed_data = format_for_analysis(input_data)
    
    print("\nOriginal data:")
    print(json.dumps(input_data, indent=2))
    
    print("\nProcessed data (what will be sent to API):")
    print(json.dumps(processed_data, indent=2))
    
    # Send to API (commented out to prevent actual API calls during testing)
    """
    try:
        response = requests.post(
            f"{base_url}/analyze-personality",
            json=processed_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nAPI response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\nAPI error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error making API request: {e}")
    """
    
    print("\nThis is a simulation. Uncomment the API call section to make actual requests.")

if __name__ == "__main__":
    test_api_with_processor()
