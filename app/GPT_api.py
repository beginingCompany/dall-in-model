import os
import re
import logging
import json
from typing import Optional, List, Union, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, ValidationError
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import tiktoken

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze-personality")

app = FastAPI()

# ----- SYSTEM PROMPT -----
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

# ----- MODELS -----
class UserRequest(BaseModel):
    id: int
    user_input: str
    new_input: Optional[str] = ""
    languages: Union[str, List[str], None] = None

    @field_validator("languages", mode="before")
    @classmethod
    def coerce_languages(cls, v):
        if v is None:
            return ["en"]
        if isinstance(v, str):
            return [v.lower().strip()]
        if isinstance(v, list):
            return [str(item).lower().strip() for item in v]
        raise ValueError("languages must be a string or list of strings")

class TraitResponse(BaseModel):
    id: int
    status: str
    description_arabic: Optional[str] = ""
    description_english: Optional[str] = ""
    missing_traits: Optional[List[str]] = None
    clarification_questions: Optional[List[str]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

# ----- HELPERS -----
def extract_json(text: str) -> str:
    pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(pattern, text.strip())
    return match.group(1) if match else text.strip()

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

def combine_inputs_safely(user_input: str, new_input: str) -> str:
    """Merge user_input and new_input cleanly."""
    if not user_input:
        return new_input
    if not new_input:
        return user_input
    return f"{user_input.strip()}\n{new_input.strip()}"

def parse_traits(user_input: str, new_input: str) -> dict:
    full_text = combine_inputs_safely(user_input, new_input).lower()
    fields = {
        "personality_traits": user_input.strip(),
        "skills": "",
    }
    # --- Try direct extraction first
    patterns = {
        "skills": r"(skills (include|are|:)?)(.+?)(\.|their|they|he|she|values|hobbies|interests|important|$)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(3).strip()
            fields[key] = val.rstrip(",")

    # --- 60-point solution: infer skills if not explicitly provided
    if not fields["skills"]:
        # Expanded keyword spotting (extend as needed)
        SKILL_KEYWORDS = [
            "technical", "hands-on", "machines", "tools", "relationship-building",
            "listening", "cooperation", "collaborat", "support", "patience", "calmness",
            "teamwork", "reliable", "compassion", "community", "problem-solving",
            "active listening", "effective", "mechanical", "practical", "skills", "abilities"
        ]
        found_skills = []
        for word in SKILL_KEYWORDS:
            # Partial matches for robustness
            if word in full_text:
                found_skills.append(word)
        # Also grab from the new_input if it contains likely skill lists
        if found_skills:
            fields["skills"] = ", ".join(sorted(set(found_skills)))
    return fields

def call_gpt(traits: dict, languages: List[str], model: str = "gpt-3.5-turbo", max_tokens: int = 1200) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    client = OpenAI(api_key=api_key)

    prompt = f"Traits: {json.dumps(traits, ensure_ascii=False)}\nlanguages: {json.dumps(languages, ensure_ascii=False)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    input_tokens = num_tokens_from_messages(messages, model=model)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        raise RuntimeError(f"OpenAI API Error: {e}")

    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    return {
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }

# ----- API ROUTE -----
@app.post("/analyze-personality", response_model=TraitResponse)
async def personality(request: Request):
    try:
        data = await request.json()
        req = UserRequest(**data)
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise HTTPException(status_code=400, detail="Invalid input JSON")

    logger.info(f"User Input: {req.user_input}")
    logger.info(f"New Input: {req.new_input}")

    parsed_traits = parse_traits(req.user_input, req.new_input)

    try:
        gpt_response = call_gpt(parsed_traits, req.languages)
        json_text = extract_json(gpt_response["content"])
        gpt_json = json.loads(json_text)
    except json.JSONDecodeError:
        logger.error(f"GPT returned invalid JSON: {gpt_response}")
        raise HTTPException(
            status_code=500,
            detail={"error": "GPT returned invalid JSON", "raw_response": gpt_response["content"]},
        )
    except Exception as e:
        logger.error(f"GPT Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    result = {
        "id": req.id,
        "input_tokens": gpt_response.get("input_tokens"),
        "output_tokens": gpt_response.get("output_tokens"),
        "total_tokens": gpt_response.get("total_tokens"),
    }

    if "description_arabic" in gpt_json or "description_english" in gpt_json:
        result.update({
            "status": "complete",
            "description_arabic": gpt_json.get("description_arabic", ""),
            "description_english": gpt_json.get("description_english", "")
        })
    elif "missing_traits" in gpt_json and "clarification_questions" in gpt_json:
        result.update({
            "status": "incomplete",
            "missing_traits": gpt_json.get("missing_traits", []),
            "clarification_questions": gpt_json.get("clarification_questions", [])
        })
    else:
        logger.error(f"Unexpected GPT output: {gpt_response}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected GPT output", "raw_response": gpt_response["content"]},
        )

    return result

# ----- ENTRY POINT -----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
