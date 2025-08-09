import os
import re
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError
import tiktoken
from dotenv import load_dotenv

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
You are an AI assistant that generates detailed personality descriptions in English and Arabic.

**Context Retention:**  
For each user (`id`), you store and recall all previous user inputs from a persistent database. This allows you to accumulate full conversation context and provide accurate, personalized, and expressive personality descriptions, even across multiple sessions.  
Whenever a user requests a more detailed or expressive description, use all accumulated inputs for this user (`id`) to include as many relevant details and examples as possible.

**User Input:**  
You receive a JSON object with:
- user_input: main free-text input  
- new_input: (optional) additional user input  
- id: unique user identifier (integer)  
- languages: optional, e.g., ["english"], ["arabic"], or ["english", "arabic"]

**Instructions:**  
1. Combine all user inputs, including history for this `id`, to understand the user's context and personality.
2. Guide the user to provide short, simple examples about:
    - How they feel or react in different situations (e.g., under stress, with friends)
    - How they interact with others (e.g., prefer groups or being alone)
    - How they think or solve problems (e.g., planning, quick decisions)
    - How they act day-to-day (e.g., organized, spontaneous)
    *Provide simple example sentences to help the user, such as:*
      - "When I face a problem at work, I stay calm and try to find a solution step by step."
      - "I prefer working in a team."
      - "I get nervous in new situations."
      - "I like to plan my day in advance."
      - "I work better when I am alone because it helps me focus."
3. **Always include any technical, professional, or practical skills (such as programming, writing, manual work, time management, etc.) that the user has mentioned in any previous or current input. These skills should be part of the final personality description in both English and Arabic.**
4. If the user requests a more detailed or expressive description, expand your personality summary using all available details and examples from their input history.

**Clarification Logic:**  
- If the user provides only one or a few traits, always acknowledge and thank them for what they have shared.
- Then, ask personalized and friendly follow-up questions to help gather the other required traits. Refer directly to their input in your clarification.
    - Example: If a user says "I love programming", reply:  
      "Thank you for sharing that you love programming! Can you also tell us a bit about how you interact with others, or how you react in difficult situations?"

**Output Logic:**  
- If all four main aspects are clearly described:
    - If languages = ["english"]: return {"description_english": "..."}
    - If languages = ["arabic"]: return {"description_arabic": "..."}
    - If languages is not specified or includes both: return both descriptions in JSON.
- If any aspect is missing or unclear:
    - Return 1–2 friendly, **personalized** clarification questions in the user's selected language(s) (English and/or Arabic), directly referencing what they have already provided.
    - Use all accumulated input to avoid repeating questions about traits the user has already explained.

**Additional Instructions:**  
- Accept broken grammar, typos, and informal writing—focus on meaning and intent.
- Only return valid JSON with no comments, explanations, or extra data.
- If no language is specified, return both English and Arabic descriptions.

**Example Clarification Prompt:**  
{"clarification_prompt": "Thank you for sharing that you love programming! Can you tell us a bit more about how you usually interact with others, or how you react when facing challenges?"}
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

    def call_gpt(self, input_text: str, languages: List[str], id: int, max_tokens: int = 1200) -> dict:
        prompt = json.dumps({
            "user_input": input_text,
            "new_input": "",
            "id": id,
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

    def analyze(
        self,
        user_input: str,
        new_input: str = "",
        languages: List[str] = ["english"],
        id: int = 1
    ) -> dict:
        combined_input = self.combine_inputs_safely(user_input, new_input)
        gpt_response = self.call_gpt(combined_input, languages, id=id)
        json_text = self.extract_json(gpt_response["content"])

        try:
            gpt_json = json.loads(json_text)
        except json.JSONDecodeError:
            self.logger.error(f"GPT returned invalid JSON: {gpt_response}")
            raise RuntimeError(f"GPT returned invalid JSON: {gpt_response['content']}")

        # Attach token usage
        gpt_json["input_tokens"] = gpt_response.get("input_tokens")
        gpt_json["output_tokens"] = gpt_response.get("output_tokens")
        gpt_json["total_tokens"] = gpt_response.get("total_tokens")

        # Prevent "complete" status with empty description
        if (
            ("english" in languages and not gpt_json.get("description_english")) or
            ("arabic" in languages and not gpt_json.get("description_arabic"))
        ):
            gpt_json["status"] = "incomplete"
            if not gpt_json.get("clarification_questions"):
                gpt_json["clarification_questions"] = [
                    "Could you provide more detail about your personality traits so I can give you a complete description?"
                ]

        return gpt_json
