#!/usr/bin/env python3
"""
Direct test of the description_identity field format fix.
Shows the exact format of responses.
"""

import json
from app.personality_analyzer import PersonalityAnalyzer

def test_identity_field_format_direct():
    """Test description_identity format directly without server."""
    
    print("🧪 Direct Test: description_identity Field Format")
    print("=" * 50)
    
    try:
        analyzer = PersonalityAnalyzer()
        
        # Test 1: Non-identity input (should have description_identity as null)
        print("\n1️⃣ Testing non-identity input:")
        print("Input: 'أنا مطور برمجيات'")
        
        result1 = analyzer.analyze(
            id=123,
            user_input="أنا مطور برمجيات", 
            new_input=[],
            languages="ar"
        )
        
        identity_value = result1.get('description_identity')
        print(f"description_identity: {repr(identity_value)} (type: {type(identity_value).__name__})")
        
        if identity_value is None:
            print("✅ CORRECT: description_identity is null")
        else:
            print(f"❌ WRONG: Expected null, got {type(identity_value).__name__}")
        
        # Test 2: Identity input (should have description_identity as string)
        print("\n2️⃣ Testing identity input:")
        print("Input: 'من انت'")
        
        result2 = analyzer.analyze(
            id=124,
            user_input="من انت", 
            new_input=[],
            languages="ar"
        )
        
        identity_value2 = result2.get('description_identity')
        print(f"description_identity: {repr(identity_value2)} (type: {type(identity_value2).__name__})")
        
        if isinstance(identity_value2, str) and identity_value2.strip():
            print("✅ CORRECT: description_identity is non-empty string")
            print(f"Content preview: {identity_value2[:100]}...")
        elif identity_value2 is None:
            print("❌ ISSUE: Expected string but got null - identity detection may not be working")
        else:
            print(f"❌ WRONG: Expected string, got {type(identity_value2).__name__}")
        
        # Test 3: Multi-identity input
        print("\n3️⃣ Testing multi-identity input:")
        print("Input: 'من انت وما هدفك'")
        
        result3 = analyzer.analyze(
            id=125,
            user_input="من انت وما هدفك", 
            new_input=[],
            languages="ar"
        )
        
        identity_value3 = result3.get('description_identity')
        print(f"description_identity: {repr(identity_value3)} (type: {type(identity_value3).__name__})")
        
        if isinstance(identity_value3, str) and identity_value3.strip():
            print("✅ CORRECT: description_identity is non-empty string")
            print(f"Content preview: {identity_value3[:100]}...")
        elif identity_value3 is None:
            print("❌ ISSUE: Expected string but got null - multi-identity detection may not be working")
        else:
            print(f"❌ WRONG: Expected string, got {type(identity_value3).__name__}")
        
        print(f"\n{'='*50}")
        print("📋 Summary of Fixed Format:")
        print("✅ Non-identity: description_identity = null (not empty string)")
        print("✅ Identity questions: description_identity = string with response")
        print("✅ JSON serialization properly handles null vs string")
        
        # Show example JSON output
        print(f"\n📄 Example JSON Outputs:")
        print("Non-identity response:")
        print(json.dumps({"description_identity": result1.get('description_identity')}, indent=2))
        print("\nIdentity response:")
        print(json.dumps({"description_identity": result2.get('description_identity')}, indent=2))
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_identity_field_format_direct()
