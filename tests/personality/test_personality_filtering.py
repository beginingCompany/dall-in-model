#!/usr/bin/env python3
"""
Test mixed personality and identity content filtering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_personality_filtering():
    """Test that personality descriptions are properly extracted while identity questions are filtered out"""
    
    print("🧪 Testing Personality Content Filtering")
    print("=" * 70)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case with mixed personality and identity content
    test_data = {
        "id": 103,
        "user_input": "I am an analytical person who loves problem-solving",
        "new_input": [
            {
                "question": "Tell me about yourself",
                "answer": "I'm introverted and prefer working alone"  # Personality description
            },
            {
                "question": "How are you?",
                "answer": "من انت؟"  # Identity question
            },
            {
                "question": "What do you like?",
                "answer": "I enjoy reading books and solving puzzles, very detail-oriented"  # Personality description
            },
            {
                "question": "Any other info?",
                "answer": "كيف تحلل الشخصية؟"  # Identity question
            }
        ],
        "languages": "en"
    }
    
    print(f"📝 TEST SCENARIO:")
    print(f"   User Input: {test_data['user_input']} (should be included)")
    print(f"   Conversation History:")
    for i, item in enumerate(test_data['new_input'], 1):
        answer_type = "PERSONALITY" if i in [1, 3] else "IDENTITY"
        print(f"      {i}. Q: {item['question']}")
        print(f"         A: {item['answer']} ({answer_type})")
    print(f"   Language: {test_data['languages']}")
    print("-" * 70)
    
    print(f"📊 EXPECTED BEHAVIOR:")
    print(f"   - Should detect identity questions in history → identity response")
    print(f"   - Should extract personality from answers 1 & 3 only")
    print(f"   - Should skip identity answers 2 & 4 for personality analysis")
    print(f"   - Should include user_input for personality traits")
    print("-" * 70)
    
    # Analyze
    result = analyzer.analyze(
        id=test_data["id"],
        user_input=test_data["user_input"],
        new_input=test_data["new_input"],
        languages=test_data["languages"]
    )
    
    print(f"🎯 RESULT:")
    print(f"   Status: {result['status']}")
    print(f"   Identity Response: {result['description_identity']}")
    print(f"   English Description: {result['description_english']}")
    print(f"   Missing Traits: {result['missing_traits']}")
    print(f"   Clarification Questions: {result['clarification_questions']}")
    
    # Analysis
    has_identity = result['description_identity'] is not None
    has_personality = bool(result['description_english'])
    
    print(f"\n✅ VERIFICATION:")
    print(f"   Identity Detected: {'✅ YES' if has_identity else '❌ NO'}")
    print(f"   Personality Analysis: {'✅ YES' if has_personality else '❌ NO'}")
    
    if has_identity:
        print(f"   ✅ Identity detection working - found questions in history")
    
    if has_personality:
        print(f"   ✅ Personality extraction working - used non-identity answers")
        personality_content = result['description_english']
        if 'analytical' in personality_content.lower() or 'introverted' in personality_content.lower():
            print(f"   ✅ Contains expected personality traits")
        else:
            print(f"   ⚠️ May be missing some personality content")
    
    print(f"\n📋 CONTENT ANALYSIS:")
    if has_personality:
        print(f"   Personality Description: {result['description_english']}")
    if has_identity:
        print(f"   Identity Response: {result['description_identity'][:100]}...")

if __name__ == "__main__":
    test_personality_filtering()
