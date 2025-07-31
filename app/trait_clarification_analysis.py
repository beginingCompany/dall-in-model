import pandas as pd
import re
import matplotlib.pyplot as plt
from config.paths import PathConfig
paths = PathConfig()
csv_path = paths.DATA_PROCESSED / "BIGINING_dataset.csv"


# --- Trait Pattern and Prompt Templates ---
TRAIT_PATTERNS = {
    "emotional": r"enthusiastic|happy|sad|calm|feel(s)?|emotion|stress|excited|passion|motivat(ed|ion)",
    "social": r"collaborative|team|help|assist|shy|introvert|extrovert|interact|polite|friendly|people|others",
    "cognitive": r"think|critical|logical|analytical|understand|reason|solve|strateg(y|ic)|intuitive",
    "behavioral": r"organized|spontaneous|routine|habit|act|impulsive|disciplined|methodical|child(ish)?"
}

CLARIFICATION_TEMPLATES = {
    "emotional": "How do you usually feel in difficult or exciting situations? What emotions come up and how do you handle them?",
    "social": "Can you describe how you typically interact with others—do you enjoy helping, leading, or prefer to work alone?",
    "cognitive": "What kind of thinking comes naturally to you? Are you analytical, imaginative, or more intuitive in decisions?",
    "behavioral": "Tell me about your habits or actions—do you prefer routines, act on impulse, or stay flexible?"
}

# --- Trait Detection ---
def detect_present_traits(text: str) -> list:
    text = str(text).lower()
    present = []
    for trait, pattern in TRAIT_PATTERNS.items():
        if re.search(pattern, text):
            present.append(trait)
    return present

# --- Clarification Prompt Generator ---
def generate_clarification_prompt(user_input: str) -> str:
    present = detect_present_traits(user_input)
    missing = [trait for trait in TRAIT_PATTERNS if trait not in present]
    if not missing:
        return ""
    elif len(missing) == 1:
        return CLARIFICATION_TEMPLATES[missing[0]]
    else:
        return " ".join(CLARIFICATION_TEMPLATES[t] for t in missing)

# --- Main Analysis Function ---
def trait_coverage_analysis(csv_path, show_plot=True, return_df=False):
    df = pd.read_csv(csv_path)
    # Combine all expected outputs for strongest trait signal
    df["combined_output"] = (
        df.get("Leadership_Motivation_en", "").fillna("") + " " +
        df.get("Emotional_Social_Intelligence_en", "").fillna("") + " " +
        df.get("Key_Strengths_Applications_en", "").fillna("")
    )
    # Analyze trait coverage per row
    trait_results = []
    for _, row in df.iterrows():
        user_input = str(row.get("description_english", ""))
        expected_output = str(row.get("combined_output", ""))
        input_traits = detect_present_traits(user_input)
        output_traits = detect_present_traits(expected_output)
        missing_traits = [trait for trait in output_traits if trait not in input_traits]
        trait_results.append({
            "user_input": user_input,
            "expected_output": expected_output,
            "input_traits": input_traits,
            "output_traits": output_traits,
            "missing_traits": missing_traits,
            "clarification_prompt": generate_clarification_prompt(user_input)
        })
    trait_df = pd.DataFrame(trait_results)
    # Global stats
    missing_summary = trait_df["missing_traits"].explode().value_counts()
    print("Most Frequently Missing Traits in User Descriptions:")
    print(missing_summary)
    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.title("Most Frequently Missing Traits In User Descriptions")
        plt.ylabel("Missing Count")
        plt.xlabel("Trait")
        missing_summary.plot(kind="bar")
        plt.show()
    print("\nTrait Comparison Results (per user):")
    print(trait_df[["user_input", "input_traits", "missing_traits", "clarification_prompt"]].head(10))
    if return_df:
        return trait_df

# --- usage ---
if __name__ == "__main__":
    trait_coverage_analysis(csv_path)
