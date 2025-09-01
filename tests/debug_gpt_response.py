"""
Debug the GPT response to see where the identity response is coming from
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer
import json

# Monkey patch to debug GPT calls
original_call_gpt = PersonalityAnalyzer.call_gpt

def debug_call_gpt(self, input_data):
    print(f"🔍 DEBUG: call_gpt called")
    print(f"   Input data: {json.dumps(input_data, indent=2)}")
    
    result = original_call_gpt(self, input_data)
    
    print(f"   GPT response content preview: {result.get('content', '')[:200]}...")
    print()
    
    return result

# Replace method
PersonalityAnalyzer.call_gpt = debug_call_gpt

def test_gpt_response():
    analyzer = PersonalityAnalyzer()
    
    print("🧪 Debug Test: GPT Response Analysis")
    print("=" * 50)
    
    result = analyzer.analyze(
        id=123,
        user_input="who are you",
        new_input=[
            {"question": "Tell me about your interests", "answer": "I like technology"},
            {"question": "How do you handle stress?", "answer": ""}
        ],
        languages="en"
    )
    
    print(f"Final result identity: {result.get('description_identity', 'None')}")

if __name__ == "__main__":
    test_gpt_response()
#!/usr/bin/env python3
"""
Debug GPT response with logging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_gpt_response():
    print("🔍 DEBUGGING GPT RESPONSE")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Override the detect_identity_question method to add logging
    original_method = analyzer.detect_identity_question
    
    def logged_detect_identity_question(text):
        print(f"📝 Input text: '{text}'")
        
        if not text:
            return False, None, None
            
        # Use GPT to classify the question
        identity_classification_prompt = f"""
Analyze this user input and determine if they are asking an identity question about the AI system/chatbot.

User input: "{text}"

Identity question categories:
1. who_are_you - asking about identity ("who are you", "tell me about yourself", "من أنت", "عرف بنفسك", etc.)
2. what_is_begining - asking about the BEGINING project ("what is begining", "ما هو بيجينينغ", "ما هو مشروع بيجينينغ", etc.)
3. purpose - asking about purpose ("why were you created", "what's your purpose", "ما هو هدفك", "لماذا تم إنشاؤك", etc.)
4. role - asking about role/function ("what do you do", "what's your role", "ما هو دورك", "ما وظيفتك", etc.)
5. developer - asking about creators ("who made you", "who's your developer", "من مطورك", "من صنعك", "من أنشأك", etc.)
6. team - asking about the team ("who's your team", "who's behind you", "من فريقك", "من وراءك", etc.)
7. understand_personality - asking about capabilities ("can you understand me", "هل تفهمني", "هل يمكنك فهم شخصيتي", etc.)
8. how_analyze - asking about methodology ("how do you work", "how do you analyze", "كيف تعمل", "كيف تحلل", etc.)
9. objectives - asking about goals ("what are your objectives", "ما أهدافك", "ما غاياتك", etc.)

Respond with ONLY ONE of these formats:
- If it's an identity question: "IDENTITY:category_name"
- If it's NOT an identity question: "NOT_IDENTITY"

Examples (English):
"who are you" -> "IDENTITY:who_are_you"
"who is ur developer" -> "IDENTITY:developer"  
"what is begining" -> "IDENTITY:what_is_begining"
"I am happy today" -> "NOT_IDENTITY"
"how do you feel" -> "NOT_IDENTITY"

Examples (Arabic):
"من أنت" -> "IDENTITY:who_are_you"
"من مطورك" -> "IDENTITY:developer"
"ما هو مشروع بيجينينغ" -> "IDENTITY:what_is_begining"
"ما هو دورك" -> "IDENTITY:role"
"أنا سعيد اليوم" -> "NOT_IDENTITY"
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert at classifying user questions about AI systems."},
                {"role": "user", "content": identity_classification_prompt}
            ]
            
            print(f"📤 Sending to GPT...")
            
            response = analyzer.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=50,
            )
            
            result = response.choices[0].message.content.strip()
            print(f"📥 GPT response: '{result}'")
            
            if result.startswith("IDENTITY:"):
                category = result.split("IDENTITY:")[1].strip()
                print(f"🎯 Extracted category: '{category}'")
                if category in analyzer.IDENTITY_RESPONSES:
                    print(f"✅ Category found in IDENTITY_RESPONSES")
                    return True, category, analyzer.IDENTITY_RESPONSES[category]
                else:
                    print(f"❌ Category '{category}' not found in IDENTITY_RESPONSES")
            else:
                print(f"ℹ️ Not an identity question")
            
            return False, None, None
            
        except Exception as e:
            print(f"💥 GPT Error: {e}")
            # Fallback to simple keyword matching if GPT fails
            print("🔄 Falling back to keyword detection...")
            return analyzer._fallback_identity_detection(text)
    
    # Test phrases
    test_phrases = [
        "who are you",
        "من مطورك",
        "من أنت"
    ]
    
    for phrase in test_phrases:
        print(f"\n{'='*50}")
        print(f"🔍 Testing: '{phrase}'")
        try:
            result = logged_detect_identity_question(phrase)
            print(f"📊 Final result: {result}")
        except Exception as e:
            print(f"💥 Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_gpt_response()
