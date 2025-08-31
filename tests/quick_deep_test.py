#!/usr/bin/env python3
"""
QUICK DEEP TEST - Core functionality only
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def quick_deep_test():
    """Quick test of core functionality"""
    analyzer = PersonalityAnalyzer()
    
    print("🚀 QUICK DEEP SYSTEM TEST")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Language Detection
    print("\n1️⃣ Testing Language Detection...")
    tests_total += 1
    try:
        arabic_detected = analyzer.detect_language("انا مهندس")
        english_detected = analyzer.detect_language("I am engineer")
        mixed_detected = analyzer.detect_language("I am احمد")
        
        if arabic_detected == "arabic" and english_detected == "english" and mixed_detected == "arabic":
            print("   ✅ Language detection working")
            tests_passed += 1
        else:
            print(f"   ❌ Language detection failed: ar={arabic_detected}, en={english_detected}, mixed={mixed_detected}")
    except Exception as e:
        print(f"   💥 Error: {e}")
    
    # Test 2: Personal Introduction
    print("\n2️⃣ Testing Personal Introduction Detection...")
    tests_total += 1
    try:
        has_intro, name, job, greeting = analyzer.detect_personal_introduction("انا المهندس احمد")
        
        if has_intro and name == "احمد" and "مهندس" in job and len(greeting) > 0:
            print("   ✅ Personal introduction detection working")
            tests_passed += 1
        else:
            print(f"   ❌ Personal introduction failed: {has_intro}, {name}, {job}")
    except Exception as e:
        print(f"   💥 Error: {e}")
    
    # Test 3: Identity vs Personality
    print("\n3️⃣ Testing Identity vs Personality...")
    tests_total += 1
    try:
        is_identity1, _, _ = analyzer.detect_identity_question("who are you")
        is_identity2, _, _ = analyzer.detect_identity_question("I am a developer")
        
        if is_identity1 and not is_identity2:
            print("   ✅ Identity vs personality classification working")
            tests_passed += 1
        else:
            print(f"   ❌ Identity classification failed: 'who are you'={is_identity1}, 'I am developer'={is_identity2}")
    except Exception as e:
        print(f"   💥 Error: {e}")
    
    # Test 4: Complete Analysis
    print("\n4️⃣ Testing Complete Analysis...")
    tests_total += 1
    try:
        test_input = {
            "id": 1,
            "user_input": "انا المهندس احمد احب العمل مع الفرق واحلل المشاكل بهدوء",
            "new_input": [],
            "languages": "ar"
        }
        
        result = analyzer.analyze(**test_input)
        response = json.loads(result["content"])
        
        has_greeting = len(response.get('personal_greeting', '').strip()) > 0
        has_status = 'status' in response
        has_arabic_desc = len(response.get('description_arabic', '').strip()) > 0
        has_name = 'احمد' in response.get('description_arabic', '') or 'أحمد' in response.get('description_arabic', '')
        
        if has_greeting and has_status and (has_arabic_desc or response.get('status') == 'incomplete'):
            print("   ✅ Complete analysis working")
            print(f"      Status: {response.get('status')}")
            print(f"      Greeting: '{response.get('personal_greeting', '')[:50]}...'")
            print(f"      Missing traits: {len(response.get('missing_traits', []))}")
            tests_passed += 1
        else:
            print(f"   ❌ Complete analysis failed")
            print(f"      Greeting: {has_greeting}")
            print(f"      Status: {has_status}")
            print(f"      Arabic desc: {has_arabic_desc}")
    except Exception as e:
        print(f"   💥 Error: {e}")
    
    # Test 5: Job-based Intelligence
    print("\n5️⃣ Testing Job-based Intelligence...")
    tests_total += 1
    try:
        # Simple engineer introduction
        simple_input = {
            "id": 2,
            "user_input": "انا المهندس احمد",
            "new_input": [],
            "languages": "ar"
        }
        
        result = analyzer.analyze(**simple_input)
        response = json.loads(result["content"])
        missing_traits = len(response.get('missing_traits', []))
        
        # Should infer some traits from "engineer", so not all 4 should be missing
        if missing_traits < 4:
            print(f"   ✅ Job-based intelligence working (missing {missing_traits}/4 traits)")
            tests_passed += 1
        else:
            print(f"   ❌ Job-based intelligence failed (missing {missing_traits}/4 traits)")
    except Exception as e:
        print(f"   💥 Error: {e}")
    
    # Test 6: Edge Cases
    print("\n6️⃣ Testing Edge Cases...")
    tests_total += 1
    try:
        edge_cases = [
            {"id": 3, "user_input": "", "new_input": [], "languages": "en"},
            {"id": 4, "user_input": "hi", "new_input": [], "languages": "en"},
        ]
        
        edge_passed = 0
        for case in edge_cases:
            result = analyzer.analyze(**case)
            response = json.loads(result["content"])
            if 'status' in response:
                edge_passed += 1
        
        if edge_passed == len(edge_cases):
            print("   ✅ Edge cases handled properly")
            tests_passed += 1
        else:
            print(f"   ❌ Edge cases failed ({edge_passed}/{len(edge_cases)} passed)")
    except Exception as e:
        print(f"   💥 Error: {e}")
    
    # Final Report
    print("\n" + "=" * 50)
    print("📊 QUICK TEST RESULTS")
    print("=" * 50)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print(f"Success Rate: {(tests_passed/tests_total)*100:.1f}%")
    
    if tests_passed == tests_total:
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("System is working excellently!")
    elif tests_passed >= tests_total * 0.8:
        print(f"\n✅ MOSTLY WORKING ({tests_passed}/{tests_total} passed)")
        print("System is in good shape with minor issues")
    else:
        print(f"\n⚠️ NEEDS ATTENTION ({tests_passed}/{tests_total} passed)")
        print("System has significant issues that need fixing")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    quick_deep_test()
