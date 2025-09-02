import json
import time
import random
from typing import Dict, Any, List
from app.personality_analyzer import PersonalityAnalyzer

class PersonalityAnalyzerTester:
    def __init__(self):
        self.analyzer = PersonalityAnalyzer()
        self.test_cases = []
        self.results = []
        self.prepare_test_cases()
    
    def prepare_test_cases(self):
        """Prepare various test cases to evaluate the analyzer's performance"""
        
        # Test Case 1: Ideal scenario - Detailed and coherent
        self.test_cases.append({
            "name": "Ideal Input - Detailed and Coherent",
            "id": 1001,
            "user_input": "I am an AI developer who loves mathematics and programming in Python. I have a background in computer science and enjoy solving complex problems.",
            "new_input": [
                {
                    "question": "How do you handle challenges in your work?",
                    "answer": "I approach challenges methodically, breaking them down into smaller parts. I enjoy the process of troubleshooting and finding elegant solutions."
                },
                {
                    "question": "How would you describe your social preferences?",
                    "answer": "I'm somewhat introverted but enjoy collaborating with small teams. I value deep discussions over small talk and prefer working with people who share my interests."
                },
                {
                    "question": "What are your daily habits and routines?",
                    "answer": "I start my day with coffee and reading tech news. I work in focused 2-hour blocks with short breaks. In the evening, I enjoy reading or playing chess to unwind."
                }
            ],
            "languages": "en"
        })
        
        # Test Case 2: Minimal information - Brief responses
        self.test_cases.append({
            "name": "Minimal Information - Brief Responses",
            "id": 1002,
            "user_input": "Programmer. Like computers.",
            "new_input": [
                {
                    "question": "Can you tell me more about yourself?",
                    "answer": "Quiet. Work a lot."
                }
            ],
            "languages": "en"
        })
        
        # Test Case 3: Emotional focus - Testing the emotional trait detection
        self.test_cases.append({
            "name": "Emotional Focus",
            "id": 1003,
            "user_input": "I feel anxious most of the time. Very sensitive to criticism and often sad. Sometimes happy when coding.",
            "new_input": [],
            "languages": "en"
        })
        
        # Test Case 4: Social focus - Testing the social trait detection
        self.test_cases.append({
            "name": "Social Focus",
            "id": 1004,
            "user_input": "I love parties and being around people. Always organizing events and talking to friends. Enjoy teamwork and collaboration.",
            "new_input": [],
            "languages": "en"
        })
        
        # Test Case 5: Cognitive focus - Testing the cognitive trait detection
        self.test_cases.append({
            "name": "Cognitive Focus",
            "id": 1005,
            "user_input": "I think analytically and solve problems logically. Strategy games are my favorite. Love puzzles and mental challenges.",
            "new_input": [],
            "languages": "en"
        })
        
        # Test Case 6: Behavioral focus - Testing the behavioral trait detection
        self.test_cases.append({
            "name": "Behavioral Focus",
            "id": 1006,
            "user_input": "Very organized. Follow strict routines. Wake up at 6am, exercise, then work until 6pm. Always plan ahead.",
            "new_input": [],
            "languages": "en"
        })
        
        # Test Case 7: Arabic language - Testing language processing
        self.test_cases.append({
            "name": "Arabic Language",
            "id": 1007,
            "user_input": "أنا مطور برمجيات أحب العمل بمفردي. أستمتع بحل المشكلات المعقدة وأقضي وقت فراغي في القراءة.",
            "new_input": [],
            "languages": "ar"
        })
        
        # Test Case 8: Both languages - Testing dual language output
        self.test_cases.append({
            "name": "Both Languages",
            "id": 1008,
            "user_input": "I am a teacher who loves helping students. I'm patient, organized, and enjoy creative activities.",
            "new_input": [],
            "languages": ["en", "ar"]
        })
        
        # Test Case 9: Contradictory information - Testing handling of inconsistencies
        self.test_cases.append({
            "name": "Contradictory Information",
            "id": 1009,
            "user_input": "I'm very outgoing and love being around people all the time.",
            "new_input": [
                {
                    "question": "How do you handle social events?",
                    "answer": "I actually hate crowds and prefer being alone. Social events drain me completely."
                }
            ],
            "languages": "en"
        })
        
        # Test Case 10: Nonsensical input - Testing robustness
        self.test_cases.append({
            "name": "Nonsensical Input",
            "id": 1010,
            "user_input": "dfhjkl asdf123 !@#$ qwerty blue dog jump moon",
            "new_input": [],
            "languages": "en"
        })

        # Test Case 11: Empty input - Testing edge case
        self.test_cases.append({
            "name": "Empty Input",
            "id": 1011,
            "user_input": "",
            "new_input": [],
            "languages": "en"
        })
        
        # Test Case 12: Very long input - Testing handling of verbose input
        long_text = "I am a software developer " + "who enjoys coding. " * 50 + "I like working with teams " + "and solving problems. " * 50
        self.test_cases.append({
            "name": "Very Long Input",
            "id": 1012,
            "user_input": long_text,
            "new_input": [],
            "languages": "en"
        })
        
    def run_tests(self):
        """Run all test cases and collect results"""
        print(f"Starting personality analyzer test suite with {len(self.test_cases)} test cases...")
        print("=" * 80)
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\nRunning test case {i+1}/{len(self.test_cases)}: {test_case['name']}")
            start_time = time.time()
            
            try:
                # Build full context from user_input and new_input
                full_context = PersonalityAnalyzer.build_full_context(
                    test_case["user_input"], 
                    test_case["new_input"]
                )
                
                # Convert languages to list format as expected by the analyzer
                languages = [test_case["languages"]] if isinstance(test_case["languages"], str) else test_case["languages"]
                
                # Call the analyzer
                result = self.analyzer.analyze(full_context, "", languages, id=test_case["id"])
                
                # Add test metadata
                result["test_case"] = test_case["name"]
                result["test_id"] = test_case["id"]
                result["success"] = True
                result["error"] = None
                result["processing_time"] = time.time() - start_time
                
                self.results.append(result)
                print(f"✓ Success: {result['status']} status in {result['processing_time']:.2f} seconds")
                
                # If there are clarification questions, print them
                if result.get("clarification_questions"):
                    print(f"- Generated {len(result['clarification_questions'])} clarification questions")
                
            except Exception as e:
                error_result = {
                    "test_case": test_case["name"],
                    "test_id": test_case["id"],
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
                self.results.append(error_result)
                print(f"✗ Error: {str(e)}")
            
            # Add a small delay between tests to avoid rate limiting
            time.sleep(1)
            
    def generate_report(self):
        """Generate a comprehensive report of test results"""
        success_count = sum(1 for r in self.results if r.get("success", False))
        complete_count = sum(1 for r in self.results if r.get("status") == "complete")
        incomplete_count = sum(1 for r in self.results if r.get("status") == "incomplete")
        error_count = sum(1 for r in self.results if not r.get("success", False))
        
        print("\n" + "=" * 80)
        print("PERSONALITY ANALYZER TEST REPORT")
        print("=" * 80)
        print(f"Total test cases: {len(self.test_cases)}")
        print(f"Successful tests: {success_count}")
        print(f"Failed tests: {error_count}")
        print(f"Complete personality profiles: {complete_count}")
        print(f"Incomplete profiles (clarification needed): {incomplete_count}")
        print("\nAVERAGE PERFORMANCE METRICS:")
        
        if success_count > 0:
            avg_processing_time = sum(r.get("processing_time", 0) for r in self.results if r.get("success", False)) / success_count
            avg_tokens = sum(r.get("total_tokens", 0) for r in self.results if r.get("success", False) and r.get("total_tokens")) / sum(1 for r in self.results if r.get("success", False) and r.get("total_tokens"))
            print(f"Average processing time: {avg_processing_time:.2f} seconds")
            print(f"Average token usage: {avg_tokens:.1f} tokens")
        
        print("\nDETAILED RESULTS:")
        for i, result in enumerate(self.results):
            print(f"\n{i+1}. {result['test_case']} (ID: {result['test_id']})")
            print(f"   Success: {result['success']}")
            
            if result['success']:
                print(f"   Status: {result.get('status', 'unknown')}")
                print(f"   Processing time: {result['processing_time']:.2f} seconds")
                print(f"   Tokens: {result.get('total_tokens', 'N/A')}")
                
                # If there are clarification questions, show the first one
                if result.get("clarification_questions"):
                    print(f"   Questions: {len(result.get('clarification_questions', []))} generated")
                    if len(result.get("clarification_questions", [])) > 0:
                        print(f"   Sample question: {result['clarification_questions'][0][:60]}...")
                
                # Show a snippet of the description if available
                if result.get("description_english"):
                    desc = result.get("description_english", "")[:80]
                    print(f"   Description snippet: {desc}..." if desc else "   No description generated")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print("\nCONCLUSION:")
        if success_count == len(self.test_cases):
            print("All tests passed successfully! The analyzer appears robust.")
        elif success_count > len(self.test_cases) * 0.8:
            print(f"Most tests ({success_count}/{len(self.test_cases)}) passed successfully. The analyzer is mostly robust but has some issues.")
        else:
            print(f"Only {success_count}/{len(self.test_cases)} tests passed successfully. The analyzer needs significant improvements.")
            
        # Recommendations based on observed issues
        print("\nRECOMMENDATIONS:")
        if error_count > 0:
            print("- Improve error handling for unexpected inputs")
        if incomplete_count > complete_count:
            print("- Adjust the threshold for considering a profile complete")
        if any(not r.get("description_english") and r.get("status") == "complete" for r in self.results if r.get("success", False)):
            print("- Fix issue with empty descriptions marked as complete")
            
        print("\nTest suite completed!")

    def run_conversation_test(self):
        """Run a multi-turn conversation test to see if the analyzer can reach complete status"""
        print("\n" + "=" * 80)
        print("CONVERSATION SIMULATION TEST")
        print("=" * 80)
        
        # Start with basic information
        test_id = 9999
        user_input = "I'm a software developer who enjoys solving complex problems."
        new_input = []
        languages = ["en", "ar"]  # Test both languages
        
        print("\nInitial Input: " + user_input)
        
        # First interaction
        full_context = PersonalityAnalyzer.build_full_context(user_input, new_input)
        result = self.analyzer.analyze(full_context, "", languages, id=test_id)
        
        print(f"\nInitial Status: {result['status']}")
        print(f"Questions generated: {len(result.get('clarification_questions', []))}")
        
        # Maximum 5 conversation turns
        for turn in range(5):
            # If we got to complete status, break the loop
            if result['status'] == "complete":
                break
                
            # Get the clarification questions
            questions = result.get('clarification_questions', [])
            if not questions:
                print("No clarification questions generated, ending conversation.")
                break
                
            print(f"\n--- Conversation Turn {turn + 1} ---")
            print(f"Question: {questions[0]}")
            
            # Generate responses based on the question content
            response = ""
            if "social" in questions[0].lower() or "interact" in questions[0].lower() or "others" in questions[0].lower():
                response = "I enjoy working in small teams where everyone has clear responsibilities. I'm somewhat introverted but can be very engaged in discussions about topics I'm passionate about."
            elif "emotion" in questions[0].lower() or "feel" in questions[0].lower() or "stress" in questions[0].lower():
                response = "I tend to stay calm under pressure. I enjoy the satisfaction of solving difficult problems, and I find coding to be meditative. When stressed, I take breaks and go for walks."
            elif "routine" in questions[0].lower() or "habit" in questions[0].lower() or "day" in questions[0].lower():
                response = "I'm very organized and follow a consistent schedule. I wake up early, exercise, then work in focused blocks with breaks. I value efficiency and planning."
            elif "think" in questions[0].lower() or "decision" in questions[0].lower() or "solve" in questions[0].lower():
                response = "I approach problems methodically, breaking them down into smaller parts. I enjoy researching different solutions before deciding on the best approach. I'm analytical but also value creative thinking."
            else:
                response = "I enjoy reading technical books and hiking on weekends. I value continuous learning and try to expand my knowledge regularly. I'm dedicated to improving my skills and staying current with technology trends."
            
            print(f"Response: {response}")
            
            # Add to new_input
            new_input.append({"question": questions[0], "answer": response})
            
            # Re-analyze
            full_context = PersonalityAnalyzer.build_full_context(user_input, new_input)
            result = self.analyzer.analyze(full_context, "", languages, id=test_id)
            
            print(f"Status: {result['status']}")
            print(f"Questions: {len(result.get('clarification_questions', []))}")
        
        # Final result
        print("\n" + "=" * 80)
        print("CONVERSATION TEST RESULTS")
        print("=" * 80)
        print(f"Conversation ended with status: {result['status']}")
        print(f"Total turns: {turn + 1}")
        
        if 'description_english' in result and result['description_english']:
            print("\nFinal English Description:")
            print(result['description_english'][:300] + "..." if len(result['description_english']) > 300 else result['description_english'])
        
        if 'description_arabic' in result and result['description_arabic']:
            print("\nFinal Arabic Description:")
            print(result['description_arabic'][:300] + "..." if len(result['description_arabic']) > 300 else result['description_arabic'])

if __name__ == "__main__":
    tester = PersonalityAnalyzerTester()
    print("1. Run all test cases")
    print("2. Run conversation simulation test")
    choice = input("Select test to run (1/2): ")
    
    if choice == "2":
        tester.run_conversation_test()
    else:
        tester.run_tests()
        tester.generate_report()
