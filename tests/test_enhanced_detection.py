#!/usr/bin/env python3
"""
Test the enhanced identity detection with smart keyword similarity.
"""

from app.personality_analyzer import PersonalityAnalyzer
import os

def test_enhanced_identity_detection():
    """Test the new 3-tier identity detection system."""
    print("🧪 Testing Enhanced Identity Detection\n")
    
    # Set dummy API key
    os.environ["OPENAI_API_KEY"] = "test_key"
    
    try:
        analyzer = PersonalityAnalyzer()
        
        # Test cases that should benefit from the enhanced approach
        test_cases = [
            # Cases that might not be in exact keywords but should be detected
            ("من طورك", "developer", "Arabic variant not in keywords"),
            ("مين بناك", "developer", "Arabic 'who built you'"),
            ("ليش موجود", "purpose", "Arabic 'why exist'"),
            ("شو شغلك", "role", "Arabic dialect 'what's your job'"),
            ("who created", "developer", "Incomplete English"),
            ("ur purpose", "purpose", "Should work normally"),
            ("من مطورك", "developer", "Should work normally"),
        ]
        
        print("🔍 Testing similarity detection:")
        for test_text, expected_category, description in test_cases:
            print(f"\n  Testing: '{test_text}' ({description})")
            
            # Test similarity checker
            similar_category = analyzer._is_similar_to_identity_keywords(test_text)
            similarity_result = "✅" if similar_category else "❌"
            print(f"    Similarity check: {similarity_result} → {similar_category}")
            
            # Test full enhanced detection (will use dummy API)
            try:
                result = analyzer.detect_identity_question(test_text)
                detected = "✅" if result[0] else "❌"
                category = result[1] if result[1] else "None"
                match = "✅" if category == expected_category else "❌"
                print(f"    Full detection: {detected} → {category} {match}")
            except Exception as e:
                print(f"    Full detection: 🔄 → Will use fallback (API failed)")
                # Test fallback
                fallback_result = analyzer._fallback_identity_detection(test_text)
                detected = "✅" if fallback_result[0] else "❌"
                category = fallback_result[1] if fallback_result[1] else "None"
                match = "✅" if category == expected_category else "❌"
                print(f"    Fallback result: {detected} → {category} {match}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    """Run enhanced detection tests."""
    print("🚀 Testing Enhanced 3-Tier Identity Detection System\n")
    print("📋 System Overview:")
    print("   1. Standard GPT classification")
    print("   2. Keyword similarity check + Enhanced GPT")
    print("   3. Direct keyword fallback")
    print()
    
    test_enhanced_identity_detection()
    
    print("\n✅ Enhanced detection testing completed!")
    print("\n📝 Benefits:")
    print("   - GPT can handle variations not in exact keywords")
    print("   - Similarity check catches edge cases like 'من طورك'")
    print("   - Enhanced prompts give GPT more context for better decisions")
    print("   - Still falls back to keywords if everything fails")

if __name__ == "__main__":
    main()
