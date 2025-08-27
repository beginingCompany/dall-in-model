#!/usr/bin/env python3
"""
Test if identity_keywords are working, especially for "مين مطورك"
"""

from app.personality_analyzer import PersonalityAnalyzer
import os

def test_identity_keywords():
    """Test the identity keywords detection."""
    print("🔍 Testing Identity Keywords Detection:")
    
    # Set dummy API key
    os.environ["OPENAI_API_KEY"] = "test_key"
    
    try:
        analyzer = PersonalityAnalyzer()
        
        # Test cases that should trigger fallback detection
        test_cases = [
            "مين مطورك",  # The problematic case
            "من مطورك",   # Should work
            "منو مطورك",  # Variant
            "ما هدفك",    # Purpose
            "ايش دورك",   # Role
            "who ur developer",  # English informal
        ]
        
        print("  Testing fallback detection directly:")
        for test_text in test_cases:
            result = analyzer._fallback_identity_detection(test_text)
            detected = "✅" if result[0] else "❌"
            category = result[1] if result[1] else "None"
            print(f"    {detected} '{test_text}' → {category}")
        
        print("\n  Testing full detection (with GPT fallback):")
        for test_text in test_cases:
            try:
                result = analyzer.detect_identity_question(test_text)
                detected = "✅" if result[0] else "❌" 
                category = result[1] if result[1] else "None"
                print(f"    {detected} '{test_text}' → {category}")
            except Exception as e:
                print(f"    🔄 '{test_text}' → Fallback triggered (GPT failed)")
                # Test fallback directly
                result = analyzer._fallback_identity_detection(test_text)
                detected = "✅" if result[0] else "❌"
                category = result[1] if result[1] else "None"
                print(f"      {detected} Fallback result → {category}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

def test_arabic_root_matching():
    """Test the advanced Arabic root-based matching."""
    print("\n🔍 Testing Arabic Root-Based Matching:")
    
    os.environ["OPENAI_API_KEY"] = "test_key"
    
    try:
        analyzer = PersonalityAnalyzer()
        
        # Test the root-based combinations
        arabic_combinations = [
            "مين طورك",      # Should match developer
            "منو صنعك",      # Should match developer  
            "ايش هدفك",      # Should match purpose
            "شو وظيفتك",     # Should match role
            "وش دورك",       # Should match role
        ]
        
        for test_text in arabic_combinations:
            result = analyzer._fallback_identity_detection(test_text)
            detected = "✅" if result[0] else "❌"
            category = result[1] if result[1] else "None"
            print(f"    {detected} '{test_text}' → {category}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing Identity Keywords Usage\n")
    
    test_identity_keywords()
    test_arabic_root_matching()
    
    print("\n✅ Keywords testing completed!")
    print("\n📝 Summary:")
    print("   - identity_keywords ARE used by the model as fallback")
    print("   - GPT is tried first, keywords are used when GPT fails")
    print("   - Enhanced Arabic matching should handle 'مين مطورك' variations")

if __name__ == "__main__":
    main()
