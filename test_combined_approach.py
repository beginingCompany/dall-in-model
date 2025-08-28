#!/usr/bin/env python3
"""
Test the new combined greeting+question approach.
This tests that greetings and questions are now integrated in one natural response.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_combined_greeting_approach():
    """Test the new combined greeting+question functionality"""
    
    analyzer = PersonalityAnalyzer()
    
    print("🔗 TESTING COMBINED GREETING+QUESTION APPROACH")
    print("="*60)
    
    # Test scenario 1: Arabic introduction with job
    print("\n1. Arabic Engineer Introduction:")
    print("Input: 'مرحبا انا وليد مهندس بيوميجات'")
    
    try:
        result = analyzer.analyze(
            id=1,
            user_input="مرحبا انا وليد مهندس بيوميجات",
            new_input=[],
            languages="ar"
        )
        
        if result and "content" in result:
            content = json.loads(result["content"])
            
            greeting = content.get('personal_greeting', '')
            questions = content.get('clarification_questions', [])
            
            print(f"✅ Status: {content.get('status', 'N/A')}")
            print(f"🎯 Combined Response: '{greeting}'")
            print(f"📋 Separate Questions: {questions}")
            
            # Check if greeting contains a question (combined approach)
            has_question_in_greeting = '?' in greeting
            has_separate_questions = len(questions) > 0 and questions[0] != ""
            
            if has_question_in_greeting and not has_separate_questions:
                print("✅ SUCCESS: Combined greeting+question approach working!")
                print("   ✓ Greeting contains question")
                print("   ✓ No separate clarification questions")
            elif has_question_in_greeting and has_separate_questions:
                print("⚠️  PARTIAL: Greeting has question but also separate questions")
            elif not has_question_in_greeting and has_separate_questions:
                print("❌ OLD APPROACH: Separate greeting and questions")
            else:
                print("❌ NO QUESTIONS: Neither combined nor separate questions found")
                
        else:
            print("❌ No valid result returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test scenario 2: English developer introduction
    print("\n2. English Developer Introduction:")
    print("Input: 'Hi I'm Sarah, I work as a developer'")
    
    try:
        result = analyzer.analyze(
            id=2,
            user_input="Hi I'm Sarah, I work as a developer",
            new_input=[],
            languages="en"
        )
        
        if result and "content" in result:
            content = json.loads(result["content"])
            
            greeting = content.get('personal_greeting', '')
            questions = content.get('clarification_questions', [])
            
            print(f"✅ Status: {content.get('status', 'N/A')}")
            print(f"🎯 Combined Response: '{greeting}'")
            print(f"📋 Separate Questions: {questions}")
            
            # Check if approach is working
            has_question_in_greeting = '?' in greeting
            has_separate_questions = len(questions) > 0 and questions[0] != ""
            
            if has_question_in_greeting and not has_separate_questions:
                print("✅ SUCCESS: Combined approach working!")
            else:
                print("⚠️  Check needed: May still be using old approach")
                
        else:
            print("❌ No valid result returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test scenario 3: Name only (should still work)
    print("\n3. Name Only Introduction:")
    print("Input: 'انا احمد'")
    
    try:
        result = analyzer.analyze(
            id=3,
            user_input="انا احمد",
            new_input=[],
            languages="ar"
        )
        
        if result and "content" in result:
            content = json.loads(result["content"])
            
            greeting = content.get('personal_greeting', '')
            questions = content.get('clarification_questions', [])
            
            print(f"✅ Status: {content.get('status', 'N/A')}")
            print(f"🎯 Combined Response: '{greeting}'")
            print(f"📋 Separate Questions: {questions}")
            
            # For name-only, we might have general questions since no job context
            if greeting:
                print("✅ Has greeting for name-only introduction")
            
        else:
            print("❌ No valid result returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_direct_method():
    """Test the combined method directly"""
    
    analyzer = PersonalityAnalyzer()
    
    print("\n" + "="*60)
    print("🔧 TESTING COMBINED METHOD DIRECTLY")
    print("="*60)
    
    test_cases = [
        {
            "name": "وليد",
            "job": "مهندس", 
            "missing_traits": ["cognitive", "social"],
            "language": "arabic"
        },
        {
            "name": "Sarah",
            "job": "developer",
            "missing_traits": ["behavioral", "emotional"],
            "language": "english"
        },
        {
            "name": "أحمد",
            "job": "",
            "missing_traits": ["cognitive"],
            "language": "arabic"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {case['name']} ({case['job'] or 'no job'})")
        
        try:
            combined_response = analyzer.generate_personalized_greeting_with_question(
                case['name'], case['job'], case['missing_traits'], case['language']
            )
            
            print(f"Combined Response: '{combined_response}'")
            
            # Check if it's properly combined
            has_greeting_elements = any(word in combined_response.lower() for word in 
                ['مرحب', 'أهلا', 'سعيد', 'hello', 'hi', 'nice', 'great'])
            has_question = '?' in combined_response
            
            if has_greeting_elements and has_question:
                print("✅ Successfully combines greeting + question")
            elif has_greeting_elements:
                print("⚠️  Has greeting but no question")
            elif has_question:
                print("⚠️  Has question but no greeting elements")
            else:
                print("❌ Neither greeting nor question detected")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_combined_greeting_approach()
    test_direct_method()
    
    print("\n" + "="*60)
    print("🎯 COMBINED GREETING+QUESTION APPROACH TESTING COMPLETE")
    print("="*60)
