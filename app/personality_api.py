import os
import json
import re
import logging
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personality_api")

# Load environment variables
load_dotenv()

app = FastAPI()

# In-memory cache (replace with a real DB for production!)
user_results_cache = {}

# Initialize OpenAI client using new SDK
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set or not loaded from .env!")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are an AI assistant building rich personality descriptions in Arabic and English.

You will receive:
- previous known traits (possibly empty)
- new user input

You must analyze all input so far, and either:

1. If info is missing to build a good personality description,
   return JSON with:
   - missing_traits: list of missing info keys
   - clarification_questions: list of questions to ask user to fill those traits

2. If all info is sufficient, return JSON with:
   - description_arabic: string
   - description_english: string

Traits to consider include:
- personality traits
- interests
- hobbies
- skills
- values

Respond only with JSON.
Never use code block formatting or triple backticks in your response.
"""

def extract_json(text):
    """
    Remove code block markers (triple backticks) and 'json' label if present.
    """
    pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(pattern, text.strip())
    if match:
        return match.group(1)
    return text.strip()

def call_gpt(previous_traits, new_input, model="gpt-3.5-turbo", max_tokens=1200):
    prompt = f"""
Previous traits: {json.dumps(previous_traits, ensure_ascii=False)}

New user input: "{new_input}"

Analyze the info and respond as instructed in the system prompt.
"""
    import time
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    duration = time.time() - start_time
    logger.info(f"OpenAI call duration: {duration:.2f} seconds")
    return response.choices[0].message.content

def cache_key(user_id, previous_traits, user_input):
    """
    Build a hashable cache key based on user id, traits, and input.
    Ensures updated traits/user_input produce a new cache entry.
    """
    return f"{user_id}:{json.dumps(previous_traits, sort_keys=True, ensure_ascii=False)}:{user_input.strip()}"

@app.post("/personality")
async def personality(request: Request):
    data = await request.json()
    user_id = data.get("id")
    user_input = data.get("user_input", "").strip()
    previous_traits = data.get("previous_traits", {})

    if not user_input:
        return JSONResponse({"error": "Missing user_input"}, status_code=400)
    if not user_id:
        return JSONResponse({"error": "Missing id"}, status_code=400)

    key = cache_key(user_id, previous_traits, user_input)
    cached_result = user_results_cache.get(key)
    if cached_result:
        logger.info(f"Cache hit for user_id={user_id}")
        return JSONResponse(cached_result)

    try:
        gpt_response = call_gpt(previous_traits, user_input)
        json_text = extract_json(gpt_response)
        gpt_json = json.loads(json_text)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from GPT: {gpt_response}")
        return JSONResponse({"error": "GPT returned invalid JSON", "raw_response": gpt_response}, status_code=500)
    except Exception as e:
        logger.error(f"Exception: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    # Prepare the result for storage and return
    result = {"id": user_id}
    if "description_arabic" in gpt_json and "description_english" in gpt_json:
        result.update({
            "status": "complete",
            "description_arabic": gpt_json["description_arabic"],
            "description_english": gpt_json["description_english"],
        })
    elif "missing_traits" in gpt_json and "clarification_questions" in gpt_json:
        result.update({
            "status": "incomplete",
            "missing_traits": gpt_json["missing_traits"],
            "clarification_questions": gpt_json["clarification_questions"],
        })
    else:
        logger.error(f"Unexpected GPT output: {gpt_response}")
        return JSONResponse({"error": "Unexpected GPT output", "raw_response": gpt_response}, status_code=500)

    user_results_cache[key] = result  # Save result for future identical requests
    logger.info(f"Cache saved for user_id={user_id}")
    return JSONResponse(result)

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)
