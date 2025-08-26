#!/usr/bin/env python3
"""
Test that ALL identity response triggers are filtered out during conversations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_all_identity_triggers_in_conversation():
    """Test that ALL identity triggers are filtered out during conversations"""
    
    print("=== Testing ALL Identity Triggers in Conversation Context ===\n")
    analyzer = PersonalityAnalyzer()
    
    # Sample conversation context
    conversation_context = [
        "Q: How do you usually interact with others in social settings?\nA: I love working in teams and taking leadership roles."
    ]
    
    # Test ALL identity triggers from IDENTITY_RESPONSES
    test_cases = [
        # who_are_you triggers
        "who are you",
        "what are you", 
        "tell me about yourself",
        "introduce yourself",
        
        # what_is_begining triggers
        "what is begining",
        "explain begining",
        "tell me about begining",
        
        # purpose triggers
        "what is your purpose",
        "why were you created",
        "why are you here",
        "what's your purpose",
        
        # role triggers
        "what is your role",
        "what do you do",
        "what's your function",
        "what's your job",
        
        # developer triggers
        "who is your developer",
        "who made you",
        "who built you",
        "who created you",
        
        # team triggers
        "who is your team",
        "who's behind you",
        "who's working with you",
        "who works with you",
        
        # understand_personality triggers
        "can you really understand personality",
        "can you analyze personality",
        "can you understand me",
        "are you able to understand",
        
        # how_analyze triggers
        "how do you work",
        "how do you analyze",
        "how does this work",
        "how do you analyze personality",
        
        # objectives triggers
        "what begining aims for",
        "what are begining objectives",
        "objectives of begining",
        "goals of begining"
    ]
    
    print(f"Testing {len(test_cases)} identity trigger phrases in conversation context...\n")
    
    passed_tests = 0
    failed_tests = 0
    
    for i, trigger in enumerate(test_cases, 1):
        print(f"Test {i}: '{trigger}'")
        
        # Test the identity response in conversation context
        identity_response = analyzer.get_identity_response(
            trigger,
            language="en",
            openai_client=None,
            conversation_context=conversation_context
        )
        
        if identity_response:
            print(f"  ❌ FAILED: Identity response triggered: {identity_response[:50]}...")
            failed_tests += 1
        else:
            print(f"  ✅ PASSED: No identity response (correctly filtered)")
            passed_tests += 1
        
        print()
    
    print("="*60)
    print("SUMMARY:")
    print(f"✅ Passed: {passed_tests}/{len(test_cases)}")
    print(f"❌ Failed: {failed_tests}/{len(test_cases)}")
    
    if failed_tests == 0:
        print("🎉 ALL IDENTITY TRIGGERS CORRECTLY FILTERED IN CONVERSATION CONTEXT!")
        print("✅ System will respond to these questions but skip trait processing")
    else:
        print("⚠️ Some identity triggers are still getting through the filter")
    
    return failed_tests == 0

def test_identity_triggers_standalone():
    """Test that identity triggers still work when asked as standalone questions"""
    
    print("\n=== Testing Identity Triggers as Standalone Questions ===\n")
    analyzer = PersonalityAnalyzer()
    
    # Test a few key triggers without conversation context
    standalone_tests = [
        "who are you",
        "what is begining", 
        "what is your purpose",
        "who is your developer"
    ]
    
    passed_tests = 0
    
    for trigger in standalone_tests:
        print(f"Testing standalone: '{trigger}'")
        
        identity_response = analyzer.get_identity_response(
            trigger,
            language="en",
            openai_client=None,
            conversation_context=None  # No conversation context
        )
        
        if identity_response:
            print(f"  ✅ PASSED: Identity response provided: {identity_response[:50]}...")
            passed_tests += 1
        else:
            print(f"  ❌ FAILED: No identity response for standalone question")
        
        print()
    
    print(f"Standalone tests: {passed_tests}/{len(standalone_tests)} passed")
    return passed_tests == len(standalone_tests)

if __name__ == "__main__":
    print("Testing comprehensive identity trigger filtering...\n")
    
    test1_passed = test_all_identity_triggers_in_conversation()
    test2_passed = test_identity_triggers_standalone()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print(f"Conversation filtering: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Standalone detection: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 ALL IDENTITY RESPONSE FILTERING IS WORKING PERFECTLY!")
        print("✅ Confused answers filtered out in conversations")
        print("✅ Genuine questions answered when standalone")
    else:
        print("⚠️ Some issues remain with identity response filtering")
