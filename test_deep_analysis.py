#!/usr/bin/env python3
"""
Deep Analysis Test Suite for Personality Analyzer
Tests all types of words, sentences, edge cases, and linguistic variations
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def test_analyzer_comprehensive():
    analyzer = PersonalityAnalyzer()
    
    # Test counters
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    def run_test(test_name, input_text, expected_category, expected_is_identity=None, expected_is_off_topic=None):
        nonlocal total_tests, passed_tests, failed_tests
        total_tests += 1
        
        print(f"\n🧪 Test {total_tests}: {test_name}")
        print(f"Input: '{input_text}'")
        
        try:
            # Test identity detection
            is_identity, identity_category, identity_response = analyzer.detect_identity_question(input_text)
            
            # Test off-topic detection
            languages = analyzer.detect_language(input_text)
            is_off_topic, off_topic_type, off_topic_response = analyzer.detect_off_topic_question(input_text, languages)
            
            # Determine actual category
            if is_identity:
                actual_category = "identity"
                detail = f"({identity_category})"
            elif is_off_topic:
                actual_category = "off_topic" 
                detail = f"({off_topic_type})"
            else:
                actual_category = "personality"
                detail = "(incomplete/normal)"
            
            print(f"Expected: {expected_category}")
            print(f"Actual: {actual_category} {detail}")
            
            # Check if test passed
            if actual_category == expected_category:
                print("✅ PASSED")
                passed_tests += 1
            else:
                print("❌ FAILED")
                failed_tests.append({
                    'test_name': test_name,
                    'input': input_text,
                    'expected': expected_category,
                    'actual': actual_category,
                    'detail': detail
                })
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed_tests.append({
                'test_name': test_name,
                'input': input_text,
                'expected': expected_category,
                'actual': f"ERROR: {e}",
                'detail': ""
            })
    
    print("🔍 COMPREHENSIVE DEEP ANALYSIS TEST SUITE")
    print("=" * 60)
    
    # ===== 1. SELF-DESCRIPTIONS (Should be personality) =====
    print("\n📝 CATEGORY 1: SELF-DESCRIPTIONS (Should be personality)")
    
    # English self-descriptions - Basic
    run_test("Basic self-description 1", "I am a developer", "personality")
    run_test("Basic self-description 2", "I'm a creative person", "personality")
    run_test("Basic self-description 3", "I work as a teacher", "personality")
    run_test("Basic self-description 4", "My job is engineering", "personality")
    run_test("Basic self-description 5", "My role is team leader", "personality")
    
    # English self-descriptions - Complex
    run_test("Complex self-description 1", "I am a software developer who enjoys creating innovative applications", "personality")
    run_test("Complex self-description 2", "I'm a creative developer working on exciting projects", "personality")
    run_test("Complex self-description 3", "My purpose in life is to help others through technology", "personality")
    run_test("Complex self-description 4", "I develop mobile applications for healthcare companies", "personality")
    run_test("Complex self-description 5", "I create software solutions that make people's lives better", "personality")
    
    # English self-descriptions - Edge cases
    run_test("Self-description with 'you'", "I am like you in many ways", "personality")
    run_test("Self-description with 'create'", "I create art in my spare time", "personality")
    run_test("Self-description with 'purpose'", "I have a purpose to serve others", "personality")
    run_test("Self-description with 'role'", "I play a role in my community", "personality")
    run_test("Self-description with 'work'", "I work hard every day", "personality")
    
    # Arabic self-descriptions - Basic
    run_test("Arabic self-description 1", "أنا مطور برمجيات", "personality")
    run_test("Arabic self-description 2", "انا مهندس", "personality")
    run_test("Arabic self-description 3", "وظيفتي في شركة تقنية", "personality")
    run_test("Arabic self-description 4", "عملي هو تطوير التطبيقات", "personality")
    run_test("Arabic self-description 5", "دوري في الفريق مهم", "personality")
    
    # Arabic self-descriptions - Complex
    run_test("Arabic complex self-description 1", "أنا مطور برمجيات أحب الإبداع", "personality")
    run_test("Arabic complex self-description 2", "أطور مواقع الويب للشركات", "personality")
    run_test("Arabic complex self-description 3", "عملي يساعد الناس في حياتهم", "personality")
    
    # Mixed language self-descriptions
    run_test("Mixed language 1", "I am مطور applications", "personality")
    run_test("Mixed language 2", "أنا developer في شركة", "personality")
    
    # ===== 2. IDENTITY QUESTIONS (Should be identity) =====
    print("\n🤖 CATEGORY 2: IDENTITY QUESTIONS (Should be identity)")
    
    # English identity questions - Basic
    run_test("Basic identity 1", "who are you", "identity")
    run_test("Basic identity 2", "what is your purpose", "identity")
    run_test("Basic identity 3", "who is your developer", "identity")
    run_test("Basic identity 4", "what do you do", "identity")
    run_test("Basic identity 5", "tell me about yourself", "identity")
    
    # English identity questions - Informal
    run_test("Informal identity 1", "who r u", "identity")
    run_test("Informal identity 2", "ur purpose", "identity")
    run_test("Informal identity 3", "ur developer", "identity")
    run_test("Informal identity 4", "what u do", "identity")
    run_test("Informal identity 5", "who ur creator", "identity")
    
    # English identity questions - Variations
    run_test("Identity variation 1", "tell me who created you", "identity")
    run_test("Identity variation 2", "what's your mission", "identity")
    run_test("Identity variation 3", "who built this system", "identity")
    run_test("Identity variation 4", "what are your objectives", "identity")
    run_test("Identity variation 5", "who's behind this project", "identity")
    
    # Arabic identity questions - Basic
    run_test("Arabic identity 1", "من أنت", "identity")
    run_test("Arabic identity 2", "ما هدفك", "identity")
    run_test("Arabic identity 3", "من مطورك", "identity")
    run_test("Arabic identity 4", "ما هو دورك", "identity")
    run_test("Arabic identity 5", "عرف بنفسك", "identity")
    
    # Arabic identity questions - Dialect variations
    run_test("Arabic dialect 1", "مين أنت", "identity")
    run_test("Arabic dialect 2", "منو أنت", "identity")
    run_test("Arabic dialect 3", "مين مطورك", "identity")
    run_test("Arabic dialect 4", "شو دورك", "identity")
    run_test("Arabic dialect 5", "ايش هدفك", "identity")
    
    # Arabic identity questions - Complex
    run_test("Arabic complex identity 1", "من هو المطور الذي صنعك", "identity")
    run_test("Arabic complex identity 2", "ما هو الهدف من إنشائك", "identity")
    run_test("Arabic complex identity 3", "أخبرني عن نفسك وعن مطوريك", "identity")
    
    # ===== 3. OFF-TOPIC QUESTIONS (Should be off_topic) =====
    print("\n🔄 CATEGORY 3: OFF-TOPIC QUESTIONS (Should be off_topic)")
    
    # Factual questions
    run_test("Factual question 1", "what color is the sky", "off_topic")
    run_test("Factual question 2", "what's the weather today", "off_topic")
    run_test("Factual question 3", "how many planets are there", "off_topic")
    run_test("Factual question 4", "what is the capital of France", "off_topic")
    
    # Technical/Learning questions
    run_test("Technical question 1", "how do I learn Python", "off_topic")
    run_test("Technical question 2", "what is machine learning", "off_topic")
    run_test("Technical question 3", "how to write better code", "off_topic")
    run_test("Technical question 4", "explain artificial intelligence", "off_topic")
    
    # General knowledge
    run_test("General knowledge 1", "tell me about history", "off_topic")
    run_test("General knowledge 2", "how does photosynthesis work", "off_topic")
    run_test("General knowledge 3", "what is quantum physics", "off_topic")
    
    # Arabic off-topic questions
    run_test("Arabic off-topic 1", "ما لون السماء", "off_topic")
    run_test("Arabic off-topic 2", "كيف أتعلم البرمجة", "off_topic")
    run_test("Arabic off-topic 3", "ما هو الطقس اليوم", "off_topic")
    run_test("Arabic off-topic 4", "أخبرني عن التاريخ", "off_topic")
    
    # ===== 4. EDGE CASES AND TRICKY SCENARIOS =====
    print("\n⚠️ CATEGORY 4: EDGE CASES AND TRICKY SCENARIOS")
    
    # Ambiguous sentences
    run_test("Ambiguous 1", "I am what I am", "personality")
    run_test("Ambiguous 2", "you are what you are", "personality")
    run_test("Ambiguous 3", "we are developers", "personality")
    run_test("Ambiguous 4", "everyone has a purpose", "personality")
    
    # Sentences with identity keywords but not identity questions
    run_test("False positive test 1", "I study developer tools", "personality")
    run_test("False positive test 2", "My friend is your typical developer", "personality")
    run_test("False positive test 3", "The purpose of this meeting", "personality")
    run_test("False positive test 4", "I work with your team sometimes", "personality")
    
    # Questions about third parties
    run_test("Third party 1", "who is the president", "off_topic")
    run_test("Third party 2", "what does Google do", "off_topic")
    run_test("Third party 3", "who created Facebook", "off_topic")
    
    # Typos and misspellings
    run_test("Typo test 1", "who ar you", "identity")
    run_test("Typo test 2", "waht is your purpose", "identity")
    run_test("Typo test 3", "I am devloper", "personality")
    run_test("Typo test 4", "my job iz programming", "personality")
    
    # Very short inputs
    run_test("Short input 1", "you", "personality")
    run_test("Short input 2", "me", "personality")
    run_test("Short input 3", "developer", "personality")
    run_test("Short input 4", "purpose", "personality")
    
    # Very long inputs
    run_test("Long input", "I am a software developer who has been working in the technology industry for many years and I really enjoy creating applications that help people solve their daily problems and make their lives easier", "personality")
    
    # Mixed content
    run_test("Mixed content 1", "I am a developer. What do you do?", "identity")  # Should detect the question part
    run_test("Mixed content 2", "Hello, who are you? I am a programmer.", "identity")  # Should detect the question part
    
    # ===== 5. LINGUISTIC VARIATIONS =====
    print("\n🌍 CATEGORY 5: LINGUISTIC VARIATIONS")
    
    # Formal vs informal
    run_test("Formal identity", "Could you please tell me about your identity?", "identity")
    run_test("Informal identity", "yo who r u", "identity")
    
    # Different sentence structures
    run_test("Inverted structure 1", "Your developer, who is he?", "identity")
    run_test("Inverted structure 2", "A developer, I am", "personality")
    
    # Questions vs statements
    run_test("Question structure", "Are you a chatbot?", "identity")
    run_test("Statement structure", "You are a chatbot", "personality")
    
    # ===== 6. SPECIAL CHARACTERS AND FORMATTING =====
    print("\n🔤 CATEGORY 6: SPECIAL CHARACTERS AND FORMATTING")
    
    # With punctuation
    run_test("With punctuation 1", "Who are you?", "identity")
    run_test("With punctuation 2", "I am a developer!", "personality")
    run_test("With punctuation 3", "What... is your purpose?", "identity")
    
    # With numbers
    run_test("With numbers 1", "I am a developer with 5 years experience", "personality")
    run_test("With numbers 2", "What are your top 3 objectives?", "identity")
    
    # With emojis (if any)
    run_test("With emojis 1", "I am a developer 😊", "personality")
    run_test("With emojis 2", "Who are you? 🤖", "identity")
    
    # Upper/lower case variations
    run_test("Uppercase", "I AM A DEVELOPER", "personality")
    run_test("Mixed case", "WhO aRe YoU", "identity")
    
    # ===== 7. CONTEXT-DEPENDENT SCENARIOS =====
    print("\n🎯 CATEGORY 7: CONTEXT-DEPENDENT SCENARIOS")
    
    # Conditional statements
    run_test("Conditional 1", "If I am a developer, then...", "personality")
    run_test("Conditional 2", "If you are AI, then who created you?", "identity")
    
    # Hypothetical scenarios  
    run_test("Hypothetical 1", "Imagine I am a developer", "personality")
    run_test("Hypothetical 2", "Suppose your purpose was different", "identity")
    
    # Comparative statements
    run_test("Comparative 1", "I am a better developer than most", "personality")
    run_test("Comparative 2", "Your purpose is clearer than mine", "identity")
    
    # ===== FINAL RESULTS =====
    print("\n" + "=" * 60)
    print("📊 DEEP ANALYSIS TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
        print("-" * 40)
        for i, test in enumerate(failed_tests, 1):
            print(f"{i}. {test['test_name']}")
            print(f"   Input: '{test['input']}'")
            print(f"   Expected: {test['expected']}")
            print(f"   Got: {test['actual']} {test['detail']}")
            print()
    else:
        print("\n🎉 ALL TESTS PASSED!")
    
    return passed_tests, total_tests, failed_tests

if __name__ == "__main__":
    test_analyzer_comprehensive()
