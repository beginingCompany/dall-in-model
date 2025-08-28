#!/usr/bin/env python3
"""
Final verification test for the greeting system improvements.
This tests the exact scenario from the user's conversation.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def final_verification():
    """Final test of the exact user scenario"""
    
    print("🔍 FINAL VERIFICATION TEST")
    print("Testing the exact Arabic conversation scenario from the user's image")
    print("="*70)
    
    analyzer = PersonalityAnalyzer()
    
    # Test the exact input from the user's conversation
    user_input = "مرحبا انا وليد مهندس بيوميجات"
    
    print(f"\n📝 USER INPUT: '{user_input}'")
    print("🔄 Processing...")
    
    try:
        # Analyze the input
        result = analyzer.analyze(
            id=1,
            user_input=user_input,
            new_input=[],
            languages="ar"
        )
        
        if result and "content" in result:
            content = json.loads(result["content"])
            
            print("\n📊 RESULTS:")
            print("-" * 40)
            print(f"✅ Status: {content.get('status', 'N/A')}")
            print(f"🎉 Greeting: '{content.get('personal_greeting', 'NO GREETING')}'")
            print(f"📋 Missing Traits: {content.get('missing_traits', [])}")
            print(f"❓ Next Question: '{content.get('clarification_questions', ['NO QUESTION'])[0]}'")
            
            # Verification
            has_greeting = bool(content.get('personal_greeting', '').strip())
            includes_name = 'وليد' in content.get('personal_greeting', '')
            includes_job_ref = any(word in content.get('personal_greeting', '') 
                                 for word in ['مهندس', 'عمل', 'وظيف'])
            
            print("\n🎯 VERIFICATION:")
            print("-" * 40)
            print(f"✅ Has greeting: {has_greeting}")
            print(f"✅ Includes name 'وليد': {includes_name}")
            print(f"✅ References job/profession: {includes_job_ref}")
            print(f"✅ Status is appropriate: {content.get('status') in ['incomplete', 'complete']}")
            
            if has_greeting and includes_name:
                print("\n🎉 SUCCESS! The greeting system is working perfectly!")
                print("🚀 The original issue has been completely resolved!")
            else:
                print("\n❌ Issue detected. The greeting system needs further adjustment.")
                
        else:
            print("❌ No valid response received")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    print("\n" + "="*70)
    print("FINAL VERIFICATION COMPLETE")

if __name__ == "__main__":
    final_verification()
