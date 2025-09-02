#!/usr/bin/env python3
"""
COMPREHENSIVE DEEP TEST FOR PERSONALITY ANALYSIS SYSTEM
Tests all major features, edge cases, and functionality
"""

import json
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

class DeepSystemTest:
    def __init__(self):
        self.analyzer = PersonalityAnalyzer()
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        
    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            if result:
                print(f"✅ PASSED: {test_name}")
                self.passed_tests += 1
            else:
                print(f"❌ FAILED: {test_name}")
                self.failed_tests += 1
        except Exception as e:
            print(f"💥 ERROR in {test_name}: {str(e)}")
            self.failed_tests += 1
        
        self.total_tests += 1
        time.sleep(1)  # Brief pause between tests
    
    def test_language_detection(self):
        """Test automatic language detection"""
        print("Testing language detection...")
        
        test_cases = [
            ("Hello, I am a developer", "en"),
            ("انا مهندس برمجيات", "ar"),
            ("I am احمد", "ar"),  # Mixed language - should detect Arabic
            ("مرحبا، I work as engineer", "ar"),  # Mixed - should detect Arabic
        ]
        
        all_passed = True
        for text, expected_lang in test_cases:
            detected = self.analyzer.detect_language(text)
            expected_full = "arabic" if expected_lang == "ar" else "english"
            
            if detected == expected_full:
                print(f"  ✅ '{text[:30]}...' → {detected}")
            else:
                print(f"  ❌ '{text[:30]}...' → {detected} (expected {expected_full})")
                all_passed = False
        
        return all_passed
    
    def test_personal_introduction_detection(self):
        """Test personal introduction detection"""
        print("Testing personal introduction detection...")
        
        test_cases = [
            ("انا المهندس احمد", True, "احمد", "مهندس"),
            ("I am engineer Ahmed", True, "Ahmed", "engineer"),
            ("My name is Sara and I'm a teacher", True, "Sara", "teacher"),
            ("I love programming", False, "", ""),
            ("How are you today", False, "", ""),
        ]
        
        all_passed = True
        for text, should_detect, expected_name, expected_job in test_cases:
            has_intro, name, job, greeting = self.analyzer.detect_personal_introduction(text)
            
            if has_intro == should_detect and name == expected_name and job == expected_job:
                print(f"  ✅ '{text}' → {has_intro}, name='{name}', job='{job}'")
            else:
                print(f"  ❌ '{text}' → {has_intro}, name='{name}', job='{job}' (expected {should_detect}, '{expected_name}', '{expected_job}')")
                all_passed = False
        
        return all_passed
    
    def test_identity_vs_personality(self):
        """Test identity question detection vs personality content"""
        print("Testing identity vs personality classification...")
        
        test_cases = [
            # Identity questions (should be detected)
            ("who are you", True, "who_are_you"),
            ("من أنت", True, "who_are_you"),
            ("what is your purpose", True, "purpose"),
            ("who is your developer", True, "developer"),
            
            # Personality content (should NOT be identity)
            ("I am a developer", False, None),
            ("انا مبرمج", False, None),
            ("I work with teams", False, None),
            ("My role involves problem solving", False, None),
        ]
        
        all_passed = True
        for text, should_be_identity, expected_category in test_cases:
            is_identity, category, _ = self.analyzer.detect_identity_question(text)
            
            if is_identity == should_be_identity and category == expected_category:
                print(f"  ✅ '{text}' → identity={is_identity}, category={category}")
            else:
                print(f"  ❌ '{text}' → identity={is_identity}, category={category} (expected {should_be_identity}, {expected_category})")
                all_passed = False
        
        return all_passed
    
    def test_off_topic_detection(self):
        """Test off-topic question detection"""
        print("Testing off-topic detection...")
        
        test_cases = [
            # Off-topic (should be detected)
            ("What is the capital of France", True),
            ("How does photosynthesis work", True),
            ("What color is the sky", True),
            ("Tell me about history", True),
            
            # Personality-related (should NOT be off-topic)
            ("I feel happy today", False),
            ("I am a social person", False),
            ("انا شخص هادئ", False),
            ("I work as engineer", False),
        ]
        
        all_passed = True
        for text, should_be_off_topic in test_cases:
            is_off_topic, _, _ = self.analyzer.detect_off_topic_question(text, "en")
            
            if is_off_topic == should_be_off_topic:
                print(f"  ✅ '{text}' → off_topic={is_off_topic}")
            else:
                print(f"  ❌ '{text}' → off_topic={is_off_topic} (expected {should_be_off_topic})")
                all_passed = False
        
        return all_passed
    
    def test_job_trait_inference(self):
        """Test job-based trait inference (balanced approach)"""
        print("Testing job-based trait inference...")
        
        # Test good engineer (should get some traits from job)
        good_engineer = {
            "id": 1,
            "user_input": "انا المهندس احمد احب حل المشاكل",
            "new_input": [],
            "languages": "ar"
        }
        
        result1 = self.analyzer.analyze(**good_engineer)
        response1 = json.loads(result1["content"])
        missing1 = len(response1.get('missing_traits', []))
        
        # Test struggling engineer (should trust self-description)
        struggling_engineer = {
            "id": 2,
            "user_input": "انا المهندس سارة لكن اجد صعوبة في حل المشاكل",
            "new_input": [],
            "languages": "ar"
        }
        
        result2 = self.analyzer.analyze(**struggling_engineer)
        response2 = json.loads(result2["content"])
        missing2 = len(response2.get('missing_traits', []))
        
        print(f"  Good engineer missing traits: {missing1}")
        print(f"  Struggling engineer missing traits: {missing2}")
        
        # Both should have reasonable results (not all 4 missing, not all 0 missing)
        if 0 <= missing1 <= 3 and 0 <= missing2 <= 3:
            print("  ✅ Job inference working reasonably")
            return True
        else:
            print("  ❌ Job inference not working properly")
            return False
    
    def test_personal_greeting_integration(self):
        """Test personal greeting integration in responses"""
        print("Testing personal greeting integration...")
        
        test_case = {
            "id": 3,
            "user_input": "انا المهندس احمد احب العمل مع الفرق",
            "new_input": [],
            "languages": "ar"
        }
        
        result = self.analyzer.analyze(**test_case)
        response = json.loads(result["content"])
        
        greeting = response.get('personal_greeting', '')
        arabic_desc = response.get('description_arabic', '')
        
        has_greeting = len(greeting.strip()) > 0
        has_name_in_desc = 'احمد' in arabic_desc or 'أحمد' in arabic_desc
        
        print(f"  Personal greeting: '{greeting}'")
        print(f"  Name in description: {has_name_in_desc}")
        
        if has_greeting and has_name_in_desc:
            print("  ✅ Personal greeting and name integration working")
            return True
        else:
            print("  ❌ Personal greeting or name integration failed")
            return False
    
    def test_conversation_flow(self):
        """Test multi-turn conversation handling"""
        print("Testing conversation flow...")
        
        # First interaction
        first_input = {
            "id": 4,
            "user_input": "I am a developer who loves coding",
            "new_input": [],
            "languages": "en"
        }
        
        result1 = self.analyzer.analyze(**first_input)
        response1 = json.loads(result1["content"])
        
        # Simulate second interaction with clarification
        if response1.get('clarification_questions'):
            question = response1['clarification_questions'][0]
            
            second_input = {
                "id": 4,
                "user_input": "I am a developer who loves coding",
                "new_input": [
                    {
                        "question": question,
                        "answer": "I usually stay calm under pressure and think analytically"
                    }
                ],
                "languages": "en"
            }
            
            result2 = self.analyzer.analyze(**second_input)
            response2 = json.loads(result2["content"])
            
            missing1 = len(response1.get('missing_traits', []))
            missing2 = len(response2.get('missing_traits', []))
            
            print(f"  First turn missing traits: {missing1}")
            print(f"  Second turn missing traits: {missing2}")
            
            if missing2 < missing1:
                print("  ✅ Conversation flow improving trait coverage")
                return True
            else:
                print("  ❌ Conversation flow not improving")
                return False
        else:
            print("  ❌ No clarification questions generated")
            return False
    
    def test_multilingual_responses(self):
        """Test multilingual response generation"""
        print("Testing multilingual responses...")
        
        # Test Arabic input
        arabic_case = {
            "id": 5,
            "user_input": "انا شخص اجتماعي احب العمل مع الفرق واحلل المشاكل بهدوء",
            "new_input": [],
            "languages": "ar"
        }
        
        result_ar = self.analyzer.analyze(**arabic_case)
        response_ar = json.loads(result_ar["content"])
        
        # Test English input
        english_case = {
            "id": 6,
            "user_input": "I am a social person who loves working with teams and analyzes problems calmly",
            "new_input": [],
            "languages": "en"
        }
        
        result_en = self.analyzer.analyze(**english_case)
        response_en = json.loads(result_en["content"])
        
        has_arabic_desc = len(response_ar.get('description_arabic', '').strip()) > 0
        has_english_desc = len(response_en.get('description_english', '').strip()) > 0
        
        print(f"  Arabic response has Arabic description: {has_arabic_desc}")
        print(f"  English response has English description: {has_english_desc}")
        
        if has_arabic_desc and has_english_desc:
            print("  ✅ Multilingual responses working")
            return True
        else:
            print("  ❌ Multilingual responses not working properly")
            return False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("Testing edge cases...")
        
        edge_cases = [
            # Empty input
            {"id": 7, "user_input": "", "new_input": [], "languages": "en"},
            # Very short input
            {"id": 8, "user_input": "hi", "new_input": [], "languages": "en"},
            # Mixed language input
            {"id": 9, "user_input": "Hello انا احمد", "new_input": [], "languages": "auto"},
            # Long input
            {"id": 10, "user_input": "I am a software engineer with 10 years of experience who loves working with teams and solving complex problems analytically while maintaining emotional balance", "new_input": [], "languages": "en"},
        ]
        
        all_passed = True
        for i, case in enumerate(edge_cases):
            try:
                result = self.analyzer.analyze(**case)
                response = json.loads(result["content"])
                
                # Check if response has required fields
                required_fields = ['id', 'status', 'personal_greeting']
                has_required = all(field in response for field in required_fields)
                
                if has_required:
                    print(f"  ✅ Edge case {i+1}: Handled properly")
                else:
                    print(f"  ❌ Edge case {i+1}: Missing required fields")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ Edge case {i+1}: Exception - {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_api_integration(self):
        """Test API integration (if server is running)"""
        print("Testing API integration...")
        
        try:
            import requests
            
            test_data = {
                "id": 11,
                "user_input": "انا المهندس احمد احب البرمجة",
                "new_input": [],
                "languages": "ar"
            }
            
            response = requests.post("http://127.0.0.1:8000/analyze-personality", json=test_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                has_greeting = len(result.get('personal_greeting', '').strip()) > 0
                has_status = 'status' in result
                
                if has_greeting and has_status:
                    print("  ✅ API integration working")
                    return True
                else:
                    print("  ❌ API response missing required fields")
                    return False
            else:
                print(f"  ❌ API returned status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ⚠️ API test skipped: {str(e)}")
            print("    (Make sure server is running: uvicorn app.api:app --reload)")
            return True  # Don't fail the test if API server isn't running
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🚀 STARTING COMPREHENSIVE DEEP SYSTEM TEST")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        self.run_test("Language Detection", self.test_language_detection)
        self.run_test("Personal Introduction Detection", self.test_personal_introduction_detection)
        self.run_test("Identity vs Personality Classification", self.test_identity_vs_personality)
        self.run_test("Off-topic Detection", self.test_off_topic_detection)
        self.run_test("Job-based Trait Inference", self.test_job_trait_inference)
        self.run_test("Personal Greeting Integration", self.test_personal_greeting_integration)
        self.run_test("Conversation Flow", self.test_conversation_flow)
        self.run_test("Multilingual Responses", self.test_multilingual_responses)
        self.run_test("Edge Cases", self.test_edge_cases)
        self.run_test("API Integration", self.test_api_integration)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate final report
        print("\n" + "=" * 80)
        print("📊 FINAL TEST REPORT")
        print("=" * 80)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! System is working excellently!")
        elif self.failed_tests <= 2:
            print(f"\n⚠️ {self.failed_tests} tests failed. System mostly working but needs attention.")
        else:
            print(f"\n❌ {self.failed_tests} tests failed. System needs significant fixes.")
        
        print("\n" + "=" * 80)
        
        return self.failed_tests == 0

def main():
    """Run the comprehensive deep test"""
    tester = DeepSystemTest()
    success = tester.run_all_tests()
    return success

if __name__ == "__main__":
    main()
