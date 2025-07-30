
import os
import re
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError
import tiktoken
from dotenv import load_dotenv
import re

load_dotenv()

class PersonalityAnalyzer:
    def generate_clarification_prompt(user_input: str) -> str:

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

        text = user_input.lower()
        present = []
        for trait, pattern in TRAIT_PATTERNS.items():
            if re.search(pattern, text):
                present.append(trait)

        missing = [trait for trait in TRAIT_PATTERNS if trait not in present]

        if not missing:
            return ""
        elif len(missing) == 1:
            return CLARIFICATION_TEMPLATES[missing[0]]
        else:
            return " ".join(CLARIFICATION_TEMPLATES[t] for t in missing)

    SYSTEM_PROMPT = """
    You are an AI assistant for generating rich personality descriptions in English and Arabic. You receive user input in JSON format with the following keys:

    - user_input: main free-text input  
    - new_input: additional input (may be from follow-up)  
    - id: unique user identifier (integer)  
    - languages: optional list like ["english"], ["arabic"], or ["english", "arabic"]

    Your primary task is to analyze the combined input and return a valid JSON response, depending on the completeness and clarity of the traits inferred.

    ---

    ### OUTPUT LOGIC

    1.If all four personality trait categories are clearly present:
    - If languages = ["english"] → return: {"description_english": "..."}
    - If languages = ["arabic"] → return: {"description_arabic": "..."}
    - If languages not specified → return: {"description_english": "...", "description_arabic": "..."}

    2. If required traits are unclear or missing:
    - Return a single intelligent clarification prompt, or a list of two related prompts, encouraging the user to expand on how they think, act, feel, and relate to others.

    ---

    ### TRAIT CATEGORIES TO EXTRACT

    Required:
    - **Emotional**: e.g., resilient, sensitive, anxious, calm
    - **Social**: e.g., extroverted, collaborative, reserved, shy
    - **Cognitive**: e.g., critical thinker, intuitive, analytical, fast learner
    - **Behavioral**: e.g., disciplined, impulsive, reactive, consistent

    Optional (if mentioned):
    - Technical skills: programming, writing, etc.
    - Interpersonal: communication, leadership, empathy
    - Practical: time management, manual skills, etc.
    - Problem-solving: decision making, adaptability

    ---

    ### CONTEXTUAL HANDLING RULES

    - Combine `user_input` and `new_input` as a single stream of input.
    - Retain user state by `id` across multiple messages and accumulate inputs.
    - Be tolerant of broken grammar, typos, or colloquial speech. Infer meaning where possible and complete sentences using linguistic context and intent.
    - If all four personality traits are still ambiguous after inference, return 1–2 polite and intelligent follow-up prompts, like:
    {"clarification_prompt": "What helps you stay focused or calm when you're under pressure?"}
    - Use polite, neutral, and natural language in prompts.
    - Avoid technical labels like “emotional trait” in prompts.

    ---

    ### OUTPUT FORMAT

    - Always return valid JSON.
    - Do NOT include any explanations, reasoning, or comments.
    - Do NOT include trailing commas or metadata.
    - If no language is specified, return both Arabic and English descriptions.
    """


    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger("PersonalityAnalyzer")

    @staticmethod
    def combine_inputs_safely(user_input: str, new_input: str) -> str:
        if not user_input:
            return new_input or ""
        if not new_input:
            return user_input
        return f"{user_input.strip()}\n{new_input.strip()}"

    @staticmethod
    def extract_json(text: str) -> str:
        pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, text.strip())
        return match.group(1) if match else text.strip()

    @staticmethod
    def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 4
        tokens_per_name = -1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def call_gpt(self, input_text: str, languages: List[str], max_tokens: int = 1200) -> dict:
        prompt = json.dumps({
            "user_input": input_text,
            "new_input": "",
            "id": 1,
            "languages": languages
        }, ensure_ascii=False)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        input_tokens = self.num_tokens_from_messages(messages, model=self.model)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except OpenAIError as e:
            self.logger.error(f"OpenAI API Error: {e}")
            raise RuntimeError(f"OpenAI API Error: {e}")

        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }

    def analyze(self, user_input: str, new_input: str = "", languages: List[str] = ["english"]) -> dict:
        combined_input = self.combine_inputs_safely(user_input, new_input)
        gpt_response = self.call_gpt(combined_input, languages)
        json_text = self.extract_json(gpt_response["content"])

        try:
            gpt_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"GPT returned invalid JSON: {gpt_response}")
            raise RuntimeError(f"GPT returned invalid JSON: {gpt_response['content']}")

        gpt_json["input_tokens"] = gpt_response.get("input_tokens")
        gpt_json["output_tokens"] = gpt_response.get("output_tokens")
        gpt_json["total_tokens"] = gpt_response.get("total_tokens")

        return gpt_json
