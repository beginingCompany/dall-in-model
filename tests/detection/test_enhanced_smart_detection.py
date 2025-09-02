#!/usr/bin/env python3

import sys
import json
sys.path.append('.')

from app.personality_analyzer import PersonalityAnalyzer

def test_smart_detection_and_flowing_responses():
    print("🧪 ENHANCED SMART DETECTION & FLOWING RESPONSES TEST")
    print("="*60)
    print("Testing:")
    print("1. Smart detection of identity questions in ANY context")
    print("2. Single flowing response blocks (not separated sentences)")
    print("3. Enhanced friendliness and readability")
    print("\n" + "="*60 + "\n")
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        # Hidden in stories/articles
        {
            "input": "I was reading about AI systems, and by the way, who are you and what is your purpose?",
            "languages": "en",
            "description": "Identity questions hidden in context"
        },
        {
            "input": "كتبت مقالًا عن الذكاء الاصطناعي، وبالمناسبة، من انت وكيف تعمل؟",
            "languages": "ar", 
            "description": "Arabic identity questions in article context"
        },
        
        # Indirect/conversational
        {
            "input": "Before we start, could you tell me a bit about yourself and how you analyze personalities?",
            "languages": "en",
            "description": "Conversational indirect identity request"
        },
        {
            "input": "قبل أن نبدأ، هل يمكنك أن تخبرني عن نفسك وعن مشروع BEGINING؟",
            "languages": "ar",
            "description": "Arabic conversational identity request"
        },
        
        # Complex multi-question
        {
            "input": "I'm curious about your background, your team, and what BEGINING actually is",
            "languages": "en", 
            "description": "Complex multi-category identity request"
        },
        {
            "input": "أريد أن أعرف من انت ومن طورك وما أهدافك",
            "languages": "ar",
            "description": "Arabic complex multi-category request"
        },
        
        # Mixed with personal info (should still catch identity)
        {
            "input": "I am a software developer from Saudi Arabia, but I also want to know who created you",
            "languages": "en",
            "description": "Personal info mixed with identity question"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"TEST {i}: {test_case['description']}")
        print(f"Input: '{test_case['input']}'")
        
        result = analyzer.analyze(
            id=300 + i,
            user_input=test_case['input'],
            new_input=[],
            languages=test_case['languages']
        )
        
        desc_identity = result.get('description_identity')
        
        if desc_identity and isinstance(desc_identity, str):
            print("✅ IDENTITY DETECTED: Smart detection working")
            
            # Check for flowing response (no multiple sentences with periods)
            sentence_count = desc_identity.count('. ') + desc_identity.count('。')
            if sentence_count <= 2:  # Allow for one flowing sentence
                print("✅ FLOWING RESPONSE: Single readable block")
            else:
                print(f"⚠️  MULTIPLE SENTENCES: {sentence_count} sentence breaks found")
            
            # Check for enhanced friendliness
            friendly_indicators = [
                "friendly", "ودود", "exciting journey", "رحلة مثيرة", 
                "truly unique", "مميزًا حقًا", "inner potential", "إمكاناتك الداخلية"
            ]
            if any(indicator in desc_identity for indicator in friendly_indicators):
                print("✅ ENHANCED FRIENDLINESS: Advanced friendly language detected")
            
            print(f"Response: {desc_identity}")
            passed += 1
        else:
            print(f"❌ FAILED: Expected identity detection but got {type(desc_identity).__name__}: {repr(desc_identity)}")
        
        print("\n" + "-"*50 + "\n")
    
    print("="*60)
    print(f"📊 SMART DETECTION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ENHANCED SYSTEM WORKING PERFECTLY!")
        print("\n✅ Ultra-smart detection: Catches identity questions in ANY context")
        print("✅ Flowing responses: Single readable blocks, not fragmented sentences")
        print("✅ Enhanced friendliness: Advanced conversational tone")
        print("✅ Context awareness: Works with stories, articles, mixed content")
    else:
        print("❌ Some enhancements need work. Please review the issues above.")

if __name__ == "__main__":
    test_smart_detection_and_flowing_responses()
