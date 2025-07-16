import os
import json
import re
import logging
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

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

Your task:
- Analyze all information received so far.

If information is missing to create a good personality description, respond with:
{
  "missing_traits": [list of missing trait keys],
  "clarification_questions": [list of clarifying questions to ask the user to fill those traits]
}

If all required information is present, respond with:
{
  "description_arabic": "string",
  "description_english": "string"
}

Traits to consider include:
- personality traits
- interests
- hobbies
- skills
- values

Respond only with JSON.
Do not use code blocks or triple backticks in your response.
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

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """
    Returns the number of tokens used by a list of messages for the OpenAI chat API.
    """
    encoding = tiktoken.encoding_for_model(model)
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
    num_tokens += 3  # every reply is primed with <im_start>assistant
    return num_tokens

def call_gpt(previous_traits, new_input, model="gpt-3.5-turbo", max_tokens=1200):
    prompt = f"""
Previous traits: {json.dumps(previous_traits, ensure_ascii=False)}

New user input: "{new_input}"

Analyze the info and respond as instructed in the system prompt.
"""
    import time
    start_time = time.time()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    # Count input tokens
    input_tokens = num_tokens_from_messages(messages, model=model)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    duration = time.time() - start_time
    logger.info(f"OpenAI call duration: {duration:.2f} seconds")

    # Output tokens from OpenAI API (safe for all current chat models)
    output_tokens = getattr(response.usage, "completion_tokens", None)
    total_tokens = getattr(response.usage, "total_tokens", None)

    return {
        "content": response.choices[0].message.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

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
        json_text = extract_json(gpt_response["content"])
        gpt_json = json.loads(json_text)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from GPT: {gpt_response}")
        return JSONResponse({"error": "GPT returned invalid JSON", "raw_response": gpt_response}, status_code=500)
    except Exception as e:
        logger.error(f"Exception: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    # Prepare the result for storage and return
    result = {"id": user_id}
    if "description_arabic" in gpt_json or "description_english" in gpt_json:
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

    # Add token usage statistics
    result.update({
        "input_tokens": gpt_response["input_tokens"],
        "output_tokens": gpt_response["output_tokens"],
        "total_tokens": gpt_response["total_tokens"],
    })

    user_results_cache[key] = result  # Save result for future identical requests
    logger.info(f"Cache saved for user_id={user_id}")
    return JSONResponse(result)

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)
