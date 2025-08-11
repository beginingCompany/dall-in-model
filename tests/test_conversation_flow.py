import requests
import json
import time
import traceback

def test_conversation_flow():
    """Test multi-turn conversation with the personality analyzer API."""
    base_url = "http://localhost:8000"
    user_id = int(time.time()) % 10000  # Generate a unique user ID based on current time
    
    print(f"Testing multi-turn conversation with user ID: {user_id}")
    
    # Initial request with minimal info
    print("\nStep 1: Initial input")
    initial_payload = {
        "id": user_id,
        "user_input": "I am a software developer",
        "languages": ["en"]
    }
    
    initial_response = make_api_call(base_url, initial_payload)
    
    # Check if we got clarification questions
    if initial_response and initial_response.get("status") == "incomplete":
        questions = initial_response.get("clarification_questions")
        if questions:
            print(f"Received questions: {questions}")
            
            # Create follow-up response with more details
            print("\nStep 2: Answering first question with detailed response")
            followup_payload = {
                "id": user_id,
                "user_input": "I am a software developer",
                "new_input": [
                    {
                        "question": questions[0],
                        "answer": """When I'm feeling stressed at work, I usually take short breaks to clear my mind and approach the problem with fresh eyes. 
                        I also prioritize tasks to focus on what's most important first. For difficult bugs, I like to talk through the problem with colleagues - 
                        explaining the issue often helps me see solutions I missed. When I'm excited about a new project or technology, 
                        I tend to dive deep into documentation and tutorials to learn as much as I can."""
                    }
                ],
                "languages": ["en"]
            }
            
            followup_response = make_api_call(base_url, followup_payload)
            
            # Check if we now have a complete profile or more questions
            if followup_response:
                print(f"Updated status: {followup_response.get('status')}")
                if followup_response.get("status") == "complete":
                    print(f"\nComplete profile generated:")
                    print(followup_response.get("description_english")[:300])
                elif followup_response.get("clarification_questions"):
                    print(f"\nNew questions received: {followup_response.get('clarification_questions')}")
                    
                    # Answer second round of questions if any
                    if followup_response.get("clarification_questions"):
                        print("\nStep 3: Answering second question")
                        second_followup_payload = {
                            "id": user_id,
                            "user_input": "I am a software developer",
                            "new_input": [
                                {
                                    "question": questions[0],
                                    "answer": """When I'm feeling stressed at work, I usually take short breaks to clear my mind and approach the problem with fresh eyes."""
                                },
                                {
                                    "question": followup_response.get("clarification_questions")[0],
                                    "answer": """I prefer to work in a structured environment with clear goals. I organize my tasks in a to-do list and track my progress. 
                                    I'm punctual with meetings and deadlines, and I like to plan my work ahead of time."""
                                }
                            ],
                            "languages": ["en"]
                        }
                        
                        second_followup_response = make_api_call(base_url, second_followup_payload)
                        
                        if second_followup_response:
                            print(f"Final status: {second_followup_response.get('status')}")
                            if second_followup_response.get("status") == "complete":
                                print(f"\nFinal complete profile:")
                                print(second_followup_response.get("description_english")[:300])
                else:
                    print("\nNo clarification questions but status is still incomplete.")
                    print(f"Debug information: {json.dumps(followup_response, indent=2)}")
        else:
            print("No clarification questions received even though status is incomplete.")
            print(f"Debug information: {json.dumps(initial_response, indent=2)}")
    else:
        print("Initial response did not request clarification as expected.")
        print(f"Debug information: {json.dumps(initial_response, indent=2)}")

def make_api_call(base_url, payload):
    """Make an API call and handle any errors"""
    try:
        print(f"Sending request to API: {json.dumps(payload, indent=2)[:200]}...")
        response = requests.post(f"{base_url}/analyze-personality", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"API response: {json.dumps(result, indent=2)[:200]}...")
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    test_conversation_flow()
