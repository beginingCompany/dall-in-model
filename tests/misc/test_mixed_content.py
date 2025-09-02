#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_mixed_content_only():
    """Test just the mixed content case that's failing"""
    print("🔍 Testing Mixed Content Detection")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # The problematic mixed content
    test_input = "I'm a data scientist who loves patterns, but I want to know how you work"
    
    print(f"📝 INPUT: {test_input}")
    print("-" * 50)
    
    result = analyzer.analyze(
        id=999,
        user_input=test_input,
        new_input=[],
        languages="en"
    )
    
    has_identity = result.get('description_identity') is not None
    
    print(f"Identity Response: {'✅ YES' if has_identity else '❌ NO'}")
    print(f"Identity Content: {result.get('description_identity', 'None')}")
    print()
    
    if has_identity:
        print("🎉 SUCCESS: Mixed content correctly detected identity question!")
        return True
    else:
        print("❌ FAILED: Mixed content not detecting identity question")
        print("The phrase 'I want to know how you work' should trigger identity detection")
        return False

if __name__ == "__main__":
    test_mixed_content_only()
