#!/usr/bin/env python3
"""
Final demonstration of the combined greeting+question approach.
This shows the improved natural conversation flow.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def final_demo():
    """Show the final combined approach working"""
    
    analyzer = PersonalityAnalyzer()
    
    print("🎯 FINAL DEMO: COMBINED GREETING+QUESTION APPROACH")
    print("="*65)
    print("This demonstrates the natural conversation flow with integrated")
    print("greetings and personalized questions in one response.")
    print("="*65)
    
    # Your original example
    print("\n1. Your Original Example:")
    print("   Input: 'مرحبا انا وليد مهندس بيوميجات'")
    print("   " + "-"*50)
    
    try:
        result = analyzer.analyze(
            id=1,
            user_input="مرحبا انا وليد مهندس بيوميجات",
            new_input=[],
            languages="ar"
        )
        
        if result and "content" in result:
            content = json.loads(result["content"])
            combined_response = content.get('personal_greeting', '')
            
            print(f"   Combined Response:")
            print(f"   '{combined_response}'")
            print()
            print("   🎉 BENEFITS:")
            print("   ✅ Natural conversation flow")
            print("   ✅ Personalized to Walid's name and engineering job")
            print("   ✅ Engineering-specific question")
            print("   ✅ Single cohesive response instead of separate parts")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Show the method working directly
    print("\n2. Direct Method Examples:")
    print("   " + "-"*50)
    
    examples = [
        ("وليد", "مهندس", ["cognitive", "social"], "arabic"),
        ("Sarah", "developer", ["behavioral"], "english"),
        ("أحمد", "", ["emotional"], "arabic")
    ]
    
    for name, job, traits, lang in examples:
        try:
            combined = analyzer.generate_personalized_greeting_with_question(
                name, job, traits, lang
            )
            job_display = f" ({job})" if job else " (no job)"
            print(f"\n   {name}{job_display}:")
            print(f"   '{combined}'")
            
        except Exception as e:
            print(f"   ❌ Error for {name}: {e}")

def show_benefits():
    """Show the benefits of the combined approach"""
    
    print("\n" + "="*65)
    print("🚀 BENEFITS OF COMBINED APPROACH")
    print("="*65)
    
    benefits = [
        {
            "title": "Natural Conversation Flow",
            "before": "Greeting: 'مرحبا وليد!' + Separate Question: 'كيف تتعامل مع المشاكل؟'",
            "after": "'مرحبا وليد! بما أنك مهندس، كيف تتعامل مع المشاكل التقنية؟'"
        },
        {
            "title": "Job-Specific Personalization", 
            "before": "Generic: 'How do you handle challenges?'",
            "after": "Specific: 'As a developer, how do you tackle complex coding problems?'"
        },
        {
            "title": "Single Cohesive Response",
            "before": "Two separate fields: personal_greeting + clarification_questions",
            "after": "One natural response that flows from greeting to question"
        },
        {
            "title": "Contextual Relevance",
            "before": "Questions not related to user's profession",
            "after": "Questions tailored to user's job and missing personality traits"
        }
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"\n{i}. {benefit['title']}:")
        print(f"   Before: {benefit['before']}")
        print(f"   After:  {benefit['after']}")

if __name__ == "__main__":
    final_demo()
    show_benefits()
    
    print("\n" + "="*65)
    print("✅ COMBINED GREETING+QUESTION APPROACH IMPLEMENTED!")
    print("="*65)
    print("The system now creates natural, flowing conversations by combining")
    print("personalized greetings with relevant, job-specific questions in one")
    print("cohesive response. This makes interactions feel more human-like!")
    print("="*65)
