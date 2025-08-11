import requests
import json
import time

def test_api():
    """Test the API with a simple request and response."""
    base_url = "http://localhost:8000"
    user_id = int(time.time()) % 10000  # Generate a unique user ID
    
    print(f"Testing with user ID: {user_id}")
    
    # Create payload with same format as your example
    payload = {
        "id": user_id,
        "user_input": "I am a software developer",
        "languages": ["en"]
    }
    
    try:
        print("\nSending initial request...")
        response = requests.post(f"{base_url}/analyze-personality", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nAPI Response:\n{json.dumps(result, indent=2)}")
            
            # If we got clarification questions, answer them
            if result.get("clarification_questions"):
                print("\nGot clarification questions, sending follow-up...")
                
                followup_payload = {
                    "id": user_id,
                    "user_input": "I am a software developer",
                    "new_input": [
                        {
                            "question": result["clarification_questions"][0],
                            "answer": "When I'm feeling stressed at work, I usually take short breaks to clear my mind and approach the problem with fresh eyes."
                        }
                    ],
                    "languages": ["en"]
                }
                
                followup_response = requests.post(f"{base_url}/analyze-personality", json=followup_payload)
                
                if followup_response.status_code == 200:
                    followup_result = followup_response.json()
                    print(f"\nFollow-up Response:\n{json.dumps(followup_result, indent=2)}")
                else:
                    print(f"Follow-up error: {followup_response.status_code} - {followup_response.text}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
