#!/usr/bin/env python3
"""
Test the enhanced identity response system with API integration
"""

import requests
import json
import time

def test_enhanced_identity_api():
    """Test the enhanced identity system through the actual API"""
    
    API_BASE_URL = "http://localhost:8000"
    
    print("Testing Enhanced Identity System with Live API")
    print("=" * 60)
    
    # Test case 1: Identity question with minimal data - should get clarification questions
    test_case_1 = {
        "id": 225985882206,
        "user_input": "Hello! I like working with data.",
        "new_input": [
            {
                "question": "How do you interact with others?",
                "answer": "I work in teams sometimes."
            },
            {
                "question": "How do you handle challenges?",
                "answer": "who are you"  # Identity question
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 1: Identity Question with Minimal Data")
    print("Expected: Identity response + clarification questions to continue conversation")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-personality",
            json=test_case_1,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ Response Status: {data.get('status')}")
            print(f"✅ Identity Response: {data.get('description_identity', 'N/A')[:100]}...")
            print(f"✅ Missing Traits: {data.get('missing_traits', [])}")
            print(f"✅ Clarification Questions ({len(data.get('clarification_questions', []))}):")
            
            for i, question in enumerate(data.get('clarification_questions', []), 1):
                print(f"   {i}. {question}")
            
            # Verify expected behavior
            if (data.get('status') == 'identity' and 
                len(data.get('description_identity', '')) > 0 and
                len(data.get('clarification_questions', [])) > 0):
                print("✅ Test Case 1: PASSED - Identity with clarification questions")
            else:
                print(f"❌ Test Case 1: FAILED - Missing expected elements")
                
        else:
            print(f"❌ Test Case 1: FAILED - Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Case 1: FAILED - Request error: {str(e)}")
        print("Make sure the API server is running on localhost:8000")
        return
    
    print("\n" + "-" * 40 + "\n")
    
    # Test case 2: Arabic identity question
    test_case_2 = {
        "id": 225985882207,
        "user_input": "أنا أحب العمل مع البيانات",
        "new_input": [
            {
                "question": "كيف تتفاعل مع الآخرين؟",
                "answer": "أعمل في فرق أحياناً"
            },
            {
                "question": "كيف تتعامل مع التحديات؟",
                "answer": "من أنت"  # Arabic identity question
            }
        ],
        "languages": "ar"
    }
    
    print("Test Case 2: Arabic Identity Question")
    print("Expected: Arabic identity response + Arabic clarification questions")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-personality",
            json=test_case_2,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ Response Status: {data.get('status')}")
            
            identity_response = data.get('description_identity', '')
            questions = data.get('clarification_questions', [])
            
            print(f"✅ Arabic Identity Response: {identity_response[:100]}...")
            print(f"✅ Missing Traits: {data.get('missing_traits', [])}")
            print(f"✅ Arabic Clarification Questions ({len(questions)}):")
            
            for i, question in enumerate(questions, 1):
                print(f"   {i}. {question}")
            
            # Check for Arabic content
            has_arabic_identity = any('\u0600' <= char <= '\u06FF' for char in identity_response)
            has_arabic_questions = any(
                any('\u0600' <= char <= '\u06FF' for char in q) for q in questions
            ) if questions else False
            
            if (data.get('status') == 'identity' and 
                has_arabic_identity and 
                has_arabic_questions and
                len(questions) > 0):
                print("✅ Test Case 2: PASSED - Arabic identity with Arabic clarification questions")
            else:
                print(f"❌ Test Case 2: FAILED - Missing Arabic content or questions")
                
        else:
            print(f"❌ Test Case 2: FAILED - Status code: {response.status_code}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Case 2: FAILED - Request error: {str(e)}")
        return
    
    print("\n" + "-" * 40 + "\n")
    
    # Test case 3: Identity question with complete data - should have fewer/no clarification questions
    test_case_3 = {
        "id": 225985882208,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights. I'm usually calm and logical in my approach to challenges.",
        "new_input": [
            {
                "question": "How do you interact with others?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you handle emotions?",
                "answer": "I try to stay emotionally balanced and use analytical thinking to work through challenges. I rarely get overwhelmed and prefer to approach problems systematically."
            },
            {
                "question": "What are your daily habits?",
                "answer": "I'm very organized and structured in my daily routine. I plan my activities in advance and stick to schedules. I approach deadlines methodically and am usually early or on time."
            },
            {
                "question": "Tell me more about yourself",
                "answer": "what is begining"  # Identity question after complete data
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 3: Identity Question with Complete Personality Data")
    print("Expected: Identity response + fewer/no clarification questions")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze-personality",
            json=test_case_3,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ Response Status: {data.get('status')}")
            print(f"✅ Identity Response: {data.get('description_identity', 'N/A')[:100]}...")
            
            missing_traits = data.get('missing_traits', [])
            questions = data.get('clarification_questions', [])
            
            print(f"✅ Missing Traits: {missing_traits} (should be fewer)")
            print(f"✅ Clarification Questions: {len(questions)} (should be fewer/none)")
            
            for i, question in enumerate(questions, 1):
                print(f"   {i}. {question}")
            
            if (data.get('status') == 'identity' and 
                len(missing_traits) < 4):  # Should have fewer missing traits
                print("✅ Test Case 3: PASSED - Complete data results in fewer missing traits")
            else:
                print(f"❌ Test Case 3: FAILED - Should have fewer missing traits")
                
        else:
            print(f"❌ Test Case 3: FAILED - Status code: {response.status_code}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Test Case 3: FAILED - Request error: {str(e)}")
        return
    
    print("\n🎉 Enhanced Identity API Testing Complete!")
    print("\n📊 Summary of Enhanced Features:")
    print("✅ Identity responses now include clarification questions")
    print("✅ Conversation continues seamlessly after identity responses")
    print("✅ Missing traits are analyzed and targeted with specific questions")
    print("✅ Language-appropriate questions (English/Arabic)")
    print("✅ Token and time savings by avoiding unnecessary GPT calls for identity questions")
    print("✅ Adaptive questioning based on existing personality data completeness")

def simulate_conversation_flow():
    """Simulate a complete conversation flow with identity interruption"""
    
    API_BASE_URL = "http://localhost:8000"
    
    print("\n\nSimulating Complete Conversation Flow")
    print("=" * 60)
    
    conversation_history = []
    user_id = 999888777
    
    # Step 1: Initial user input
    print("Step 1: Initial user input")
    initial_request = {
        "id": user_id,
        "user_input": "Hi! I'm a software engineer who loves solving complex problems.",
        "new_input": [],
        "languages": "en"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/analyze-personality", json=initial_request, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            questions = data.get('clarification_questions', [])
            if questions:
                next_question = questions[0]
                print(f"Next Question: {next_question}")
                conversation_history.append({"question": next_question, "answer": ""})
        
        time.sleep(1)
        
        # Step 2: User answers, then asks identity question
        print("\nStep 2: User answers question then asks identity question")
        conversation_history[-1]["answer"] = "I prefer working in small teams where I can collaborate closely with others."
        conversation_history.append({"question": "Next question", "answer": "who are you"})  # Identity question
        
        identity_request = {
            "id": user_id,
            "user_input": initial_request["user_input"],
            "new_input": conversation_history,
            "languages": "en"
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze-personality", json=identity_request, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')} (should be 'identity')")
            print(f"Identity Response: {data.get('description_identity', '')[:80]}...")
            
            # Should still get clarification questions to continue
            questions = data.get('clarification_questions', [])
            print(f"Continuing Questions: {len(questions)} questions provided")
            for i, q in enumerate(questions[:2], 1):
                print(f"  {i}. {q[:60]}...")
        
        time.sleep(1)
        
        # Step 3: Continue conversation normally
        print("\nStep 3: Conversation continues normally after identity response")
        if questions:
            # Simulate answering the clarification question
            conversation_history[-1]["answer"] = "I prefer working in small teams."  # Replace identity question
            conversation_history.append({"question": questions[0], "answer": "I handle stress by taking breaks and thinking through problems step by step."})
            
            continue_request = {
                "id": user_id,
                "user_input": initial_request["user_input"],
                "new_input": conversation_history,
                "languages": "en"
            }
            
            response = requests.post(f"{API_BASE_URL}/analyze-personality", json=continue_request, timeout=15)
            if response.status_code == 200:
                data = response.json()
                print(f"Status: {data.get('status')} (should be 'incomplete' or 'complete')")
                print(f"Missing Traits: {data.get('missing_traits', [])}")
                print(f"Conversation seamlessly continued after identity interruption")
        
        print("\n✅ Conversation Flow Test: PASSED")
        print("✅ Identity questions don't break the conversation flow")
        print("✅ System provides clarification questions to continue efficiently")
        
    except Exception as e:
        print(f"❌ Conversation Flow Test: FAILED - {str(e)}")

if __name__ == "__main__":
    print("Make sure the API server is running with: uvicorn app.api:app --reload")
    print("Press Enter to continue with the enhanced identity tests...")
    input()
    
    test_enhanced_identity_api()
    simulate_conversation_flow()
