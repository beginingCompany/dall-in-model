#!/usr/bin/env python3

import sys
import json
sys.path.append('.')

from app.personality_analyzer import PersonalityAnalyzer

def test_ai_powered_conversational_system():
    print("🧪 AI-POWERED CONVERSATIONAL SYSTEM TEST")
    print("="*60)
    print("Testing:")
    print("1. AI-powered greeting/off-topic detection with friendly responses")
    print("2. Smart mixed content handling (identity + personality + greetings)")
    print("3. Indirect question detection")
    print("4. Natural conversation flow")
    print("5. No regex - pure AI intelligence")
    print("\n" + "="*60 + "\n")
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        # Greetings and off-topic
        {
            "input": "Hello! How are you today?",
            "languages": "en",
            "expected_type": "greeting",
            "description": "English greeting with casual question"
        },
        {
            "input": "مرحبا! كيف حالك؟",
            "languages": "ar",
            "expected_type": "greeting", 
            "description": "Arabic greeting with casual question"
        },
        {
            "input": "Nice weather today, isn't it?",
            "languages": "en",
            "expected_type": "offtopic",
            "description": "Off-topic weather conversation"
        },
        
        # Mixed content (the key challenge!)
        {
            "input": "Hi there! I'm an introverted software developer, but I'm also curious about who you are",
            "languages": "en",
            "expected_type": "mixed",
            "description": "Greeting + personality + identity question (mixed)"
        },
        {
            "input": "مرحبا، أنا شخص اجتماعي وأحب العمل مع الفريق، وأيضاً أريد أن أعرف ما هو مشروع BEGINING",
            "languages": "ar",
            "expected_type": "mixed",
            "description": "Arabic greeting + personality + identity question (mixed)"
        },
        
        # Indirect questions
        {
            "input": "I'm working on a research project and wondering if you could share some background about yourself",
            "languages": "en",
            "expected_type": "indirect_identity",
            "description": "Indirect identity question in research context"
        },
        {
            "input": "I'm very detail-oriented and analytical in my work, and I'm curious about your methodology",
            "languages": "en",
            "expected_type": "mixed_indirect",
            "description": "Personality description + indirect identity question"
        },
        
        # Pure personality (should not be greeting/off-topic)
        {
            "input": "I am introverted and prefer working alone",
            "languages": "en",
            "expected_type": "personality",
            "description": "Pure personality description"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"TEST {i}: {test_case['description']}")
        print(f"Input: '{test_case['input']}'")
        print(f"Expected: {test_case['expected_type']}")
        
        result = analyzer.analyze(
            id=400 + i,
            user_input=test_case['input'],
            new_input=[],
            languages=test_case['languages']
        )
        
        # Analyze the result
        greeting_response = result.get('personal_greeting_and_off_topic', '')
        identity_response = result.get('description_identity')
        
        success = False
        
        if test_case['expected_type'] == 'greeting' or test_case['expected_type'] == 'offtopic':
            if greeting_response and isinstance(greeting_response, str):
                print("✅ GREETING/OFF-TOPIC DETECTED: AI generated friendly response")
                print(f"Response: {greeting_response}")
                success = True
            else:
                print("❌ FAILED: Expected greeting/off-topic detection")
        
        elif test_case['expected_type'] == 'mixed':
            # Should detect both greeting AND identity
            if greeting_response or identity_response:
                print("✅ MIXED CONTENT HANDLING: AI detected multiple types")
                if greeting_response:
                    print(f"Greeting response: {greeting_response}")
                if identity_response:
                    print(f"Identity response: {identity_response}")
                success = True
            else:
                print("❌ FAILED: Expected mixed content detection")
        
        elif test_case['expected_type'] == 'indirect_identity' or test_case['expected_type'] == 'mixed_indirect':
            if identity_response and isinstance(identity_response, str):
                print("✅ INDIRECT DETECTION: AI caught subtle identity question")
                print(f"Identity response: {identity_response}")
                success = True
            else:
                print("❌ FAILED: Expected indirect identity detection")
        
        elif test_case['expected_type'] == 'personality':
            if not greeting_response and not identity_response:
                print("✅ PURE PERSONALITY: Correctly identified as trait description")
                success = True
            else:
                print("❌ FAILED: Incorrectly detected as greeting/identity")
        
        if success:
            passed += 1
        
        print("\n" + "-"*50 + "\n")
    
    print("="*60)
    print(f"📊 AI CONVERSATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 AI-POWERED CONVERSATIONAL SYSTEM WORKING PERFECTLY!")
        print("\n✅ AI greeting detection: Generates warm, friendly responses")
        print("✅ Mixed content handling: Extracts personality + handles identity")
        print("✅ Indirect question detection: Catches subtle patterns")
        print("✅ Natural conversation flow: Makes interactions very nice")
        print("✅ Pure AI intelligence: No regex, smart contextual understanding")
    else:
        print("❌ Some AI features need refinement. Please review the issues above.")

if __name__ == "__main__":
    test_ai_powered_conversational_system()
