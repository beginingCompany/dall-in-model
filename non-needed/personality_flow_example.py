import json

def process_personality_input(input_data, model_function):
    """
    input_data: dict with 'user_input', 'new_input', etc.
    model_function: callable that sends data to your personality analysis model
    """
    # Step 1: Send to model
    first_pass = model_function(input_data)
    # Step 2: If incomplete, ask missing questions
    if first_pass.get("status") == "incomplete":
        print("\n[System] Missing traits detected:", first_pass["missing_traits"])
        answers = {}
        for q in first_pass["clarification_questions"]:
            answers[q] = input(f"{q} ")
        # Step 3: Merge answers into new_input
        input_data.setdefault("new_input", [])
        for question, answer in answers.items():
            input_data["new_input"].append({"question": question, "answer": answer})
        # Step 4: Re-run analysis
        return model_function(input_data)
    return first_pass

