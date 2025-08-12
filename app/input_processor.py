"""
Question processing utility module for handling AI questions and user responses.
"""

def process_inputs(user_input, new_input):
    """
    Process the initial user input and the question-answer pairs from new_input.
    
    Args:
        user_input (str): The initial user input text
        new_input (list): List of question-answer dictionaries
        
    Returns:
        dict: A dictionary containing processed data
    """
    # Start with the initial user input
    result = {
        "initial_input": user_input,
        "combined_user_input": user_input,  # Will contain user input + all answers (no questions)
        "qa_pairs": []
    }
    
    if not new_input or not isinstance(new_input, list):
        return result
    
    # Process each question-answer pair
    for item in new_input:
        if not isinstance(item, dict):
            continue
            
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        
        if not question or not answer:
            continue
            
        # Handle multiple questions in a single string (split by comma or question mark)
        sub_questions = []
        if '?' in question:
            # Split by question mark and clean up
            sub_qs = [q.strip() + '?' for q in question.split('?') if q.strip()]
            sub_questions.extend(sub_qs)
        elif ',' in question:
            # Split by comma
            sub_questions = [q.strip() for q in question.split(',') if q.strip()]
        else:
            sub_questions = [question]
            
        # Add each question-answer to the result
        for sq in sub_questions:
            if sq:  # Check if the question is not empty
                result["qa_pairs"].append({
                    "question": sq,
                    "answer": answer
                })
                
        # Append ONLY THE ANSWER to combined_user_input (no questions)
        # Make sure there's proper separation from previous content
        if not result["combined_user_input"].endswith('.') and not result["combined_user_input"].endswith('\n'):
            result["combined_user_input"] += ". "
        else:
            result["combined_user_input"] += " "
            
        result["combined_user_input"] += answer
        
    return result


def format_for_analysis(user_data):
    """
    Format the user data for personality analysis.
    
    Args:
        user_data (dict): User data including id, user_input, new_input, and languages
        
    Returns:
        dict: Formatted data ready for analysis
    """
    # Extract the main components
    user_id = user_data.get("id")
    user_input = user_data.get("user_input", "")
    new_input = user_data.get("new_input", [])
    languages = user_data.get("languages", "en")
    
    # Process the inputs
    processed = process_inputs(user_input, new_input)
    
    # Create the formatted result
    result = {
        "id": user_id,
        "user_input": processed["combined_user_input"],  # Send user input + answers as user_input
        "new_input": processed["qa_pairs"],              # Send the structured Q&A pairs
        "languages": languages
    }
    
    return result
