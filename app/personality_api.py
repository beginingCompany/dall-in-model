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
logger = logging.getLogger("personality_api")

app = FastAPI()

# ----- SYSTEM PROMPT -----
SYSTEM_PROMPT = """
You are an AI assistant for building rich personality descriptions in Arabic and English.

You receive a JSON object with keys: personality_traits, interests, hobbies, skills, values.

You must generate a JSON response with either:
- If all required traits are present, respond only with:
  {"description_arabic": "...", "description_english": "..."}
- If any required trait is missing, respond only with:
  {"missing_traits": [list of missing trait keys], "clarification_questions": [one clarifying question per missing key]}
- If user input is empty or unparseable, request all traits as missing.

If user provides a preferred language as a "languages" or "language" field, generate only the requested description.

Traits to consider: personality_traits, interests, hobbies, skills, values

Respond only with valid, minimal JSON (no comments, no trailing commas, use double quotes).
"""

# ----- INPUT MODEL -----
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

# ----- OUTPUT MODEL -----
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
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        tokens_per_message = 4
        tokens_per_name = -1
    else:
        raise NotImplementedError(f"Token counting not implemented for model: {model}")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def parse_traits(user_input: str, new_input: str) -> dict:
    """
    Simple heuristic parser for extracting fields from new_input/user_input.
    For best accuracy, pass each trait as a clearly labeled sentence in new_input.
    """
    text = f"{user_input or ''}\n{new_input or ''}".lower()
    fields = {
        "personality_traits": user_input.strip(),
        "interests": "",
        "hobbies": "",
        "skills": "",
        "values": ""
    }
    patterns = {
        "interests": r"(interests? (include|are|:)?)(.+?)(\.|their|they|he|she|skills|values|hobbies|important|$)",
        "hobbies": r"(hobbies (include|are|:)?)(.+?)(\.|their|they|he|she|skills|values|interests|important|$)",
        "skills": r"(skills (include|are|:)?)(.+?)(\.|their|they|he|she|values|hobbies|interests|important|$)",
        "values": r"(values (include|are|:)?|important values for this individual are)(.+?)(\.|their|they|he|she|skills|hobbies|interests|important|$)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(3).strip()
            if val.endswith(","):
                val = val[:-1]
            fields[key] = val
    return fields

def call_gpt(traits: dict, languages: List[str], model: str = "gpt-3.5-turbo", max_tokens: int = 1200) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set or not loaded from .env!")
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
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    return {
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

@app.post("/personality", response_model=TraitResponse)
async def personality(request: Request):
    try:
        data = await request.json()
        req = UserRequest(**data)
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        logger.error(f"Exception parsing input: {e}")
        raise HTTPException(status_code=400, detail="Invalid input JSON")

    # Parse all traits from user_input and new_input
    parsed_traits = parse_traits(req.user_input, req.new_input)

    try:
        gpt_response = call_gpt(
            traits=parsed_traits,
            languages=req.languages
        )
        json_text = extract_json(gpt_response["content"])
        gpt_json = json.loads(json_text)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from GPT: {gpt_response}")
        raise HTTPException(
            status_code=500,
            detail={"error": "GPT returned invalid JSON", "raw_response": gpt_response["content"]},
        )
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Build response
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
