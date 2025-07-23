import os
import re
import logging
import json
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError
import tiktoken
from dotenv import load_dotenv
load_dotenv()

class PersonalityAnalyzer:
    SYSTEM_PROMPT = """
    You are an AI assistant for building rich personality descriptions in Arabic or English as backend request.

    You receive a JSON object with keys: personality_traits, skills

    You must generate a JSON response with either:
    - If all required traits are present, respond only with one language:
      {"description_arabic": "...", "description_english": "..."}
    - If any required trait is missing, respond only with:
      {"missing_traits": [list of missing trait keys], "clarification_questions": [one clarifying question per missing key]}
    - If user input is empty or unparseable, request all traits as missing.

    If user provides a preferred language as a "languages" or "language" field, generate only the requested description.

    Traits to consider: personality_traits, skills

    Respond only with valid, minimal JSON (no comments, no trailing commas, use double quotes).
    """

    SKILL_KEYWORDS = [
        "technical", "hands-on", "machines", "tools", "relationship-building",
        "listening", "cooperation", "collaborat", "support", "patience", "calmness",
        "teamwork", "reliable", "compassion", "community", "problem-solving",
        "active listening", "effective", "mechanical", "practical", "skills", "abilities"
    ]

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
            return new_input
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

    def parse_traits(self, user_input: str, new_input: str) -> dict:
        full_text = self.combine_inputs_safely(user_input, new_input).lower()
        fields = {
            "personality_traits": user_input.strip(),
            "skills": "",
        }
        # Direct extraction
        patterns = {
            "skills": r"(skills (include|are|:)?)(.+?)(\.|their|they|he|she|values|hobbies|interests|important|$)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(3).strip()
                fields[key] = val.rstrip(",")
        # 60-point: keyword spotting if not present
        if not fields["skills"]:
            found_skills = [w for w in self.SKILL_KEYWORDS if w in full_text]
            if found_skills:
                fields["skills"] = ", ".join(sorted(set(found_skills)))
        return fields

    def call_gpt(self, traits: dict, languages: List[str], max_tokens: int = 1200) -> dict:
        prompt = f"Traits: {json.dumps(traits, ensure_ascii=False)}\nlanguages: {json.dumps(languages, ensure_ascii=False)}"
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

    def analyze(self, user_input: str, new_input: str, languages: List[str]) -> dict:
        traits = self.parse_traits(user_input, new_input)
        gpt_response = self.call_gpt(traits, languages)
        json_text = self.extract_json(gpt_response["content"])
        try:
            gpt_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"GPT returned invalid JSON: {gpt_response}")
            raise RuntimeError(f"GPT returned invalid JSON: {gpt_response['content']}")
        # Merge token info
        gpt_json["input_tokens"] = gpt_response.get("input_tokens")
        gpt_json["output_tokens"] = gpt_response.get("output_tokens")
        gpt_json["total_tokens"] = gpt_response.get("total_tokens")
        return gpt_json
