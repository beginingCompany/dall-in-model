#!/usr/bin/env python3
"""
Test the specific scenario from the Arabic conversation in the user's image.
This simulates the exact conversation flow to ensure proper greeting handling.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_arabic_conversation_scenario():
    """Test the exact scenario from the user's conversation"""
    
    analyzer = PersonalityAnalyzer()
    
    print("Testing Arabic Conversation Scenario")
    print("=" * 50)
    
    # First message: "مرحبا انا وليد مهندس بيوميجات"
    print("\n1. User introduces themselves:")
    print("Input: 'مرحبا انا وليد مهندس بيوميجات'")
    
    try:
        result1 = analyzer.analyze(
            id=1,
            user_input="مرحبا انا وليد مهندس بيوميجات",
            new_input=[],
            languages="ar"
        )
        
        if result1 and "content" in result1:
            content1 = json.loads(result1["content"])
            print(f"Status: {content1.get('status', 'N/A')}")
            print(f"Personal Greeting: '{content1.get('personal_greeting', '')}'")
            print(f"Missing Traits: {content1.get('missing_traits', [])}")
            
            first_question = content1.get('clarification_questions', [''])[0]
            print(f"AI's Question: '{first_question}'")
            
            if content1.get('personal_greeting'):
                print("✅ GREETING DISPLAYED - User should see a friendly welcome!")
            else:
                print("❌ NO GREETING - This is the issue we're fixing!")
                
        else:
            print("❌ No valid result returned")
            return
            
    except Exception as e:
        print(f"❌ Error in first analysis: {e}")
        return
    
    # Second message: User asks "من أنت" (Who are you?)
    print("\n2. User asks 'من أنت' (Who are you?):")
    
    try:
        result2 = analyzer.analyze(
            id=1,
            user_input="مرحبا انا وليد مهندس بيوميجات",
            new_input=[
                {"question": first_question, "answer": "من أنت"}
            ],
            languages="ar"
        )
        
        if result2 and "content" in result2:
            content2 = json.loads(result2["content"])
            print(f"Status: {content2.get('status', 'N/A')}")
            print(f"Personal Greeting: '{content2.get('personal_greeting', '')}'")
            
            if content2.get('status') == 'identity':
                print(f"Identity Response: '{content2.get('description_identity', '')[:100]}...'")
                print("✅ IDENTITY QUESTION HANDLED PROPERLY")
            else:
                print("❌ Identity question not detected properly")
                
        else:
            print("❌ No valid result returned")
            
    except Exception as e:
        print(f"❌ Error in second analysis: {e}")
    
    # Third message: User says "أفضل الافتراب" (I prefer to approach)
    print("\n3. User responds with 'أفضل الافتراب':")
    
    try:
        # Get a behavioral question first
        behavioral_question = "كيف تتصرف عادة في البيئات الجماعية مقابل التفاعلات الفردية؟"
        
        result3 = analyzer.analyze(
            id=1,
            user_input="مرحبا انا وليد مهندس بيوميجات",
            new_input=[
                {"question": first_question, "answer": "من أنت"},
                {"question": behavioral_question, "answer": "أفضل الافتراب"}
            ],
            languages="ar"
        )
        
        if result3 and "content" in result3:
            content3 = json.loads(result3["content"])
            print(f"Status: {content3.get('status', 'N/A')}")
            print(f"Personal Greeting: '{content3.get('personal_greeting', '')}'")
            print(f"Missing Traits: {content3.get('missing_traits', [])}")
            
            next_question = content3.get('clarification_questions', [''])[0]
            print(f"AI's Next Question: '{next_question}'")
            
            # In this case, there should be no new greeting since it's a follow-up
            if not content3.get('personal_greeting'):
                print("✅ NO DUPLICATE GREETING - Good conversation flow!")
            else:
                print("⚠️  Greeting present in follow-up (might be expected)")
                
        else:
            print("❌ No valid result returned")
            
    except Exception as e:
        print(f"❌ Error in third analysis: {e}")

def test_greeting_timing():
    """Test when greetings should and shouldn't appear"""
    
    analyzer = PersonalityAnalyzer()
    
    print("\n\nTesting Greeting Timing Control")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "First introduction - should have greeting",
            "user_input": "انا سارة مطورة",
            "new_input": [],
            "should_have_greeting": True
        },
        {
            "name": "Identity question - no greeting needed",
            "user_input": "انا سارة مطورة", 
            "new_input": [{"question": "كيف تشعر في المواقف الصعبة؟", "answer": "ما هو مشروع BEGINING؟"}],
            "should_have_greeting": False
        },
        {
            "name": "Personality answer - no new greeting",
            "user_input": "انا سارة مطورة",
            "new_input": [{"question": "كيف تشعر في المواقف الصعبة؟", "answer": "أشعر بالثقة والهدوء"}],
            "should_have_greeting": False
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        
        try:
            result = analyzer.analyze(
                id=2,
                user_input=scenario['user_input'],
                new_input=scenario['new_input'],
                languages="ar"
            )
            
            if result and "content" in result:
                content = json.loads(result["content"])
                has_greeting = bool(content.get('personal_greeting', '').strip())
                
                print(f"Has greeting: {has_greeting}")
                print(f"Greeting: '{content.get('personal_greeting', '')}'")
                
                if has_greeting == scenario['should_have_greeting']:
                    print("✅ Greeting timing is correct!")
                else:
                    print(f"❌ Greeting timing is wrong! Expected: {scenario['should_have_greeting']}, Got: {has_greeting}")
                    
            else:
                print("❌ No valid result returned")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_arabic_conversation_scenario()
    test_greeting_timing()
