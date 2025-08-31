#!/usr/bin/env python3
"""
Integration test for the identity response system with the API
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_full_integration():
    """Test the full integration including API format"""
    
    print("Testing Full Identity System Integration")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case 1: Identity question in English
    test_case_1 = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles."
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 1: Identity Question in English")
    print(f"Input: {json.dumps(test_case_1, indent=2)}")
    
    try:
        result = analyzer.analyze(
            id=test_case_1["id"],
            user_input=test_case_1["user_input"],
            new_input=test_case_1["new_input"],
            languages=test_case_1["languages"]
        )
        
        # Parse the JSON response
        response_data = json.loads(result["content"])
        
        print(f"Response Status: {response_data.get('status')}")
        print(f"Identity Response: {response_data.get('description_identity', '')[:150]}...")
        print(f"English Description: {response_data.get('description_english', 'EMPTY')}")
        print(f"Arabic Description: {response_data.get('description_arabic', 'EMPTY')}")
        print(f"Missing Traits: {response_data.get('missing_traits', [])}")
        print(f"Clarification Questions: {response_data.get('clarification_questions', [])}")
        
        # Verify expected behavior
        assert response_data["status"] == "identity", f"Expected 'identity', got '{response_data['status']}'"
        assert "description_identity" in response_data, "Missing description_identity field"
        assert len(response_data["description_identity"]) > 0, "Empty identity description"
        assert response_data["description_english"] == "", "English description should be empty for identity responses"
        assert response_data["description_arabic"] == "", "Arabic description should be empty for identity responses"
        
        print("✅ Test Case 1: PASSED")
        
    except Exception as e:
        print(f"❌ Test Case 1: FAILED - {str(e)}")
        return
    
    print("\n" + "-" * 50 + "\n")
    
    # Test case 2: Identity question in Arabic
    test_case_2 = {
        "id": 225985882207,
        "user_input": "أنا شخص يحب العمل مع البيانات",
        "new_input": [
            {
                "question": "كيف تتفاعل مع الآخرين؟",
                "answer": "من أنت"
            }
        ],
        "languages": "ar"
    }
    
    print("Test Case 2: Identity Question in Arabic")
    print(f"Input: {json.dumps(test_case_2, indent=2, ensure_ascii=False)}")
    
    try:
        result = analyzer.analyze(
            id=test_case_2["id"],
            user_input=test_case_2["user_input"],
            new_input=test_case_2["new_input"],
            languages=test_case_2["languages"]
        )
        
        # Parse the JSON response
        response_data = json.loads(result["content"])
        
        print(f"Response Status: {response_data.get('status')}")
        print(f"Identity Response: {response_data.get('description_identity', '')[:150]}...")
        
        # Verify expected behavior
        assert response_data["status"] == "identity", f"Expected 'identity', got '{response_data['status']}'"
        assert "description_identity" in response_data, "Missing description_identity field"
        assert len(response_data["description_identity"]) > 0, "Empty identity description"
        # Check that it's in Arabic
        assert any('\u0600' <= char <= '\u06FF' for char in response_data["description_identity"]), "Expected Arabic text in identity response"
        
        print("✅ Test Case 2: PASSED")
        
    except Exception as e:
        print(f"❌ Test Case 2: FAILED - {str(e)}")
        return
    
    print("\n" + "-" * 50 + "\n")
    
    # Test case 3: Non-identity question (should proceed normally)
    test_case_3 = {
        "id": 225985882208,
        "user_input": "Hello! I'm someone who really enjoys working with data.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and mentoring colleagues."
            },
            {
                "question": "How do you handle emotions?",
                "answer": "I analyze problems systematically to solve them."
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 3: Non-Identity Question (Normal Processing)")
    print(f"Input: {json.dumps(test_case_3, indent=2)}")
    
    try:
        result = analyzer.analyze(
            id=test_case_3["id"],
            user_input=test_case_3["user_input"],
            new_input=test_case_3["new_input"],
            languages=test_case_3["languages"]
        )
        
        # This should be a normal GPT response (may contain various statuses)
        print(f"Response Type: {'GPT Response' if 'content' in result else 'Direct Response'}")
        print(f"Content Preview: {str(result)[:200]}...")
        
        # For non-identity questions, we should get a normal GPT response
        assert "content" in result, "Expected normal GPT response with 'content' field"
        
        print("✅ Test Case 3: PASSED (Normal processing)")
        
    except Exception as e:
        print(f"❌ Test Case 3: FAILED - {str(e)}")
        return
    
    print("\n🎉 All integration tests passed!")

def test_specific_identity_responses():
    """Test specific identity response categories"""
    
    print("\n\nTesting Specific Identity Response Categories")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    identity_tests = [
        ("who are you", "English - Who are you"),
        ("tell me about you", "English - Tell me about you"),
        ("what is begining", "English - What is BEGINING"),
        ("who is your developer", "English - Developer question"),
        ("من أنت", "Arabic - Who are you"),
        ("ما هو BEGINING", "Arabic - What is BEGINING"),
    ]
    
    for question, description in identity_tests:
        print(f"\nTesting: {description}")
        print(f"Question: '{question}'")
        
        test_input = {
            "id": 12345,
            "user_input": "Test input",
            "new_input": [{"question": "Test", "answer": question}],
            "languages": "ar" if any('\u0600' <= char <= '\u06FF' for char in question) else "en"
        }
        
        try:
            result = analyzer.analyze(
                id=test_input["id"],
                user_input=test_input["user_input"],
                new_input=test_input["new_input"],
                languages=test_input["languages"]
            )
            
            response_data = json.loads(result["content"])
            
            assert response_data["status"] == "identity", f"Expected 'identity' status"
            identity_response = response_data.get("description_identity", "")
            assert len(identity_response) > 0, "Expected non-empty identity response"
            
            print(f"✅ Response: {identity_response[:100]}...")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")

if __name__ == "__main__":
    test_full_integration()
    test_specific_identity_responses()
