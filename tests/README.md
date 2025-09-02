# Tests Organization

This document describes the organization of test files in the `/tests` directory.

## Directory Structure

### `/tests/priority_logic/`

Tests related to conversation priority logic and latest answer handling:

- `test_restructured_logic.py` - Tests for restructured conversation logic
- `test_latest_answer_priority.py` - Tests for latest answer priority over user_input
- `test_comprehensive_priority.py` - Comprehensive priority scenario testing

### `/tests/off_topic/`

Tests for off-topic content detection and response generation:

- `test_off_topic_fix.py` - Tests for off-topic classification fixes
- `test_improved_offtopic.py` - Tests for improved off-topic responses suitable for concatenation
- `test_identity_offtopic.py` - Tests for identity vs off-topic detection

### `/tests/scenarios/`

Specific scenario testing and edge cases:

- `test_exact_scenario.py` - Tests for exact user-reported scenarios
- `test_with_comma.py` - Tests with specific formatting (comma) scenarios

### `/tests/api/`

API endpoint testing:

- Various API integration tests

### `/tests/identity/`

Identity question detection and response tests:

- Tests for Arabic and English identity questions
- Context-aware identity detection
- Comprehensive identity triggers

### `/tests/conversation/`

Conversation flow and context handling tests

### `/tests/detection/`

General detection algorithm tests

### `/tests/greeting/`

Greeting detection and response tests

### `/tests/language/`

Language-specific testing (Arabic/English)

### `/tests/personality/`

Personality trait analysis and detection tests

### `/tests/integration/`

End-to-end integration tests

### `/tests/utilities/`

Utility function and analyzer component tests:

- `test_direct_analyzer.py` - Direct analyzer testing
- `test_organization_summary_root.py` - Organization summary utilities
- Other analyzer utility tests

### `/tests/misc/`

Miscellaneous tests:

- `test_new_logic.py` - Various logic testing
- Other miscellaneous tests

### `/tests/edge_cases/`

Edge case and boundary condition tests

## Running Tests

To run tests by category:

```bash
# Priority logic tests
python -m pytest tests/priority_logic/

# Off-topic tests  
python -m pytest tests/off_topic/

# Scenario tests
python -m pytest tests/scenarios/

# All tests
python -m pytest tests/
```

## Recent Additions

The following tests were recently moved from the root directory:

- Priority logic tests (latest answer prioritization)
- Off-topic response improvements
- Specific scenario debugging tests
- Utility function tests

These tests focus on the latest improvements to conversation flow, off-topic handling, and priority logic that ensures the latest answer in conversation history takes precedence over user_input for classification.
