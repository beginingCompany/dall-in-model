#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_improved_arabic_patterns():
    """Test the improved Arabic patterns with various dialectal variations"""
    
    print("=" * 80)
    print("IMPROVED ARABIC PATTERNS TEST")
    print("=" * 80)
    
    # Test the specific case that was failing
    failing_case = "مين الي مطورك"
    print(f"\n🎯 TESTING FAILING CASE: '{failing_case}'")
    response = PersonalityAnalyzer.get_identity_response(failing_case, "ar")
    if response:
        print(f"✅ NOW WORKING: {response[:50]}...")
    else:
        print("❌ Still not working")
    
    # Test comprehensive Arabic variations
    arabic_tests = {
        "who_are_you": [
            "من أنت",
            "مين انت", 
            "مين أنت",
            "انت مين",
            "قولي مين انت",
            "من انت بالضبط",
            "مين انت بالضبط",
            "عرفني على نفسك",
            "احكيلي عنك"
        ],
        "developer": [
            "من هو مطورك",
            "مين مطورك", 
            "مين الي مطورك",
            "مين اللي مطورك",
            "من صنعك",
            "مين صنعك",
            "مين الي صنعك",
            "من طورك",
            "مين طورك",
            "من عملك",
            "مين عملك"
        ],
        "purpose": [
            "ما هو هدفك",
            "ايش هدفك",
            "شو هدفك", 
            "ليش انت هنا",
            "ما غرضك",
            "ايش غرضك",
            "احكيلي عن هدفك",
            "ليش اتصنعت",
            "لماذا انت هنا"
        ],
        "role": [
            "ما هو دورك",
            "ايش دورك",
            "شو دورك",
            "شو بتعمل",
            "ايش شغلك",
            "شو شغلك",
            "بتشتغل ايش",
            "دورك ايش",
            "وظيفتك ايش"
        ],
        "what_is_begining": [
            "ما هو BEGINING",
            "ايش BEGINING",
            "شو هو BEGINING",
            "شرحلي BEGINING",
            "احكيلي عن BEGINING",
            "ما معنى BEGINING",
            "ايش يعني BEGINING",
            "BEGINING يعني ايش"
        ],
        "team": [
            "من هو فريقك",
            "مين فريقك",
            "شو فريقك",
            "فريقك مين",
            "مين الي معك",
            "مين اللي معك",
            "مين الي يشتغل معك",
            "من يعمل معك",
            "مين زملاؤك"
        ],
        "how_analyze": [
            "كيف تعمل",
            "كيف تحلل",
            "شلون تعمل",
            "شلون تحلل",
            "ايش طريقتك",
            "شو طريقتك",
            "كيف يشتغل تحليلك",
            "بأي طريقة تحلل",
            "آلية عملك"
        ],
        "understand_personality": [
            "هل يمكنك فهم شخصيتي",
            "تقدر تحللني",
            "بتفهمني",
            "هل تفهم الشخصية",
            "ممكن تحللني",
            "بتعرف تحلل الشخصية",
            "هل انت دقيق",
            "مدى دقتك"
        ],
        "objectives": [
            "ما هي أهداف BEGINING",
            "اهداف BEGINING",
            "أهداف BEGINING",
            "ما أهداف BEGINING",
            "ايش أهداف BEGINING",
            "شو أهداف BEGINING",
            "غايات المشروع",
            "ما غايات BEGINING"
        ]
    }
    
    total_tests = 0
    passed_tests = 0
    
    for category, tests in arabic_tests.items():
        print(f"\n📝 TESTING {category.upper()} - Arabic Variations:")
        print("-" * 60)
        
        for test_input in tests:
            total_tests += 1
            response = PersonalityAnalyzer.get_identity_response(test_input, "ar")
            
            if response:
                print(f"  ✅ '{test_input}' -> {response[:40]}...")
                passed_tests += 1
            else:
                print(f"  ❌ '{test_input}' -> No response")
    
    print(f"\n" + "=" * 80)
    print("ARABIC PATTERNS SUMMARY")
    print("=" * 80)
    print(f"Total Arabic Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    # Test mixed language scenarios
    print(f"\n" + "=" * 80)
    print("MIXED LANGUAGE SCENARIOS")
    print("=" * 80)
    
    mixed_tests = [
        ("مين developer تبعك", "Mixed Arabic-English"),
        ("who is مطورك", "Mixed English-Arabic"), 
        ("ما هو your purpose", "Mixed Arabic-English"),
        ("كيف تعمل analysis", "Mixed Arabic-English")
    ]
    
    for test_input, description in mixed_tests:
        response = PersonalityAnalyzer.get_identity_response(test_input, "ar")
        print(f"📝 {description}: '{test_input}'")
        if response:
            print(f"  ✅ Response: {response[:50]}...")
        else:
            print(f"  ❌ No response")
        print()

if __name__ == "__main__":
    test_improved_arabic_patterns()
