#!/usr/bin/env python3
"""
Debug the user example
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_user_example():
    analyzer = PersonalityAnalyzer()

    test_data = {
        'id': 225985882206,
        'user_input': 'who is ur developer',
        'new_input': [],
        'languages': 'en'
    }

    print('=== DEBUGGING USER EXAMPLE ===')
    user_input = test_data['user_input']
    print(f'Input: {user_input}')

    # Test detection step by step
    is_detected, category, response_data = analyzer.detect_identity_question(user_input)
    print(f'Detection result: {is_detected}, {category}')

    if response_data:
        expected_response = analyzer.get_identity_response(response_data, test_data['languages'])
        print(f'Expected response: {expected_response[:100]}...')
    else:
        print('No response_data returned')

    # Run full analysis
    result = analyzer.analyze(**test_data)
    response = json.loads(result['content'])
    
    print(f'Actual status: {response.get("status")}')
    actual_response = response.get('description_identity', 'None')
    print(f'Actual response: {actual_response}')
    missing_traits = response.get('missing_traits')
    print(f'Missing traits: {missing_traits}')
    
    # Check if the problem is in the fallback
    print('\n=== TESTING FALLBACK ===')
    fallback_result = analyzer._fallback_identity_detection(user_input)
    print(f'Fallback result: {fallback_result}')

if __name__ == "__main__":
    debug_user_example()
