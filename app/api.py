import time
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import Optional, Union, List
from app.predict import PersonalityPredictor
from app.personality_analyzer import PersonalityAnalyzer
from app.input_processor import format_for_analysis

# Get the directory of the current file
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="BEGINING Personality Analysis API",
    description="API for personality trait analysis and classification using the BEGINING system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

predictor = PersonalityPredictor(num_labels=120, top_k=3)
analyzer = PersonalityAnalyzer()

# Simple in-memory user memory storage
_user_memory_store = {}

class PredictionRequest(BaseModel):
    text: str

class PredictionItem(BaseModel):
    class_name: str
    confidence: str

class PredictionResponse(BaseModel):
    text: str
    predictions: List[PredictionItem]

class UserRequest(BaseModel):
    id: int
    user_input: str
    new_input: List[dict] = []  # Each dict: {"question": str, "answer": str}
    languages: Union[str, List[str], None] = None

    @classmethod
    def coerce_languages(cls, v):
        if v is None:
            return ["en"]
        if isinstance(v, str):
            return [v.lower().strip()]
        if isinstance(v, list):
            return [str(item).lower().strip() for item in v]
        raise ValueError("languages must be a string or list of strings")

    def get_languages(self):
        return self.coerce_languages(self.languages)

    def get_combined_new_input(self) -> str:
        """
        Combine all 'answer' fields from new_input list into a single string.
        """
        if not self.new_input:
            return ""
        # Accept both old and new formats for backward compatibility
        if isinstance(self.new_input, list) and all(isinstance(item, dict) and ("answer" in item) for item in self.new_input):
            return "\n".join(str(item.get("answer", "")).strip() for item in self.new_input if item.get("answer"))
        return str(self.new_input)

class TraitResponse(BaseModel):
    id: int
    status: str
    personal_greeting: Optional[str] = ""  # New field for friendly greetings
    description_arabic: Optional[str] = ""
    description_english: Optional[str] = ""
    description_identity: Optional[str] = ""  # Field for identity responses
    description_off_topic: Optional[str] = ""  # Field for off-topic responses
    missing_traits: Optional[List[str]] = []
    clarification_questions: Optional[List[str]] = []
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

# In-memory functions to simulate DB
def load_user_memory(user_id: int):
    return _user_memory_store.get(user_id, {"user_input": "", "new_input": ""})

def save_user_memory(user_id: int, user_input: str, new_input: str):
    _user_memory_store[user_id] = {"user_input": user_input, "new_input": new_input}

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "BEGINING Personality Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "analyze_personality": "/analyze-personality",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if analyzer and predictor are loaded
        test_result = analyzer.detect_language("test")
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "analyzer": "loaded",
                "predictor": "loaded"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        df_result = predictor.predict([request.text])
        result = df_result.iloc[0]
        preds = [
            PredictionItem(
                class_name=p.get("class_name") or p.get("class") or p.get("label", ""),
                confidence=p.get("confidence", "0%")
            )
            for p in result['predictions']
        ]
        return PredictionResponse(text=result['text'], predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/analyze-personality", response_model=TraitResponse)
async def analyze_personality(request: Request):
    t0 = time.time()
    try:
        data = await request.json()
        
        # Process the input data using our formatter
        formatted_data = format_for_analysis(data)
        
        # Then validate with Pydantic
        req = UserRequest(**formatted_data)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input JSON")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request error: {str(e)}")

    # Update memory (optional, keep if needed)
    memory = load_user_memory(req.id)
    memory["user_input"] = (memory.get("user_input", "") + " " + req.user_input.strip()).strip()
    memory["new_input"] = (memory.get("new_input", "") + " " + req.get_combined_new_input().strip()).strip()
    save_user_memory(req.id, memory["user_input"], memory["new_input"])

    # Log the input for debugging
    print(f"Processing analyze-personality request for user ID: {req.id}")
    print(f"User input (length: {len(req.user_input)}): {req.user_input[:100]}...")
    print(f"New input items: {len(req.new_input)}")
    print(f"Languages: {req.languages}")
            
    # Prepare input for analyzer
    try:
        gpt_json = analyzer.analyze(
            id=req.id,
            user_input=req.user_input,
            new_input=req.new_input,
            languages=req.languages
        )
        print(f"Analyzer returned response with keys: {list(gpt_json.keys())}")
    except Exception as e:
        print(f"ANALYZER ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Return a user-friendly response instead of an error
        return TraitResponse(
            id=req.id,
            status="incomplete",
            personal_greeting="",
            description_english="",
            description_arabic="",
            description_identity="",
            description_off_topic="",
            missing_traits=["emotional", "social", "cognitive", "behavioral"],
            clarification_questions=[
                "Could you provide more specific details about your personality?",
                "How would you describe your typical emotional responses?",
                "What are your typical behaviors in different situations?",
                "How do you interact with others in social settings?"
            ]
        )

    # Parse the model output from gpt_json["content"]
    import json
    try:
        if "content" not in gpt_json or not gpt_json["content"]:
            raise ValueError("No content in analyzer response")
        
        print(f"Parsing content (length: {len(gpt_json['content'])}): {gpt_json['content'][:100]}...")
        model_output = json.loads(gpt_json["content"])
        print("Successfully parsed JSON content")
        
    except json.JSONDecodeError as je:
        print(f"JSON DECODE ERROR: {str(je)}")
        # Provide a meaningful fallback response
        model_output = {
            "id": req.id,
            "status": "incomplete",
            "personal_greeting": "",
            "description_english": "",
            "description_arabic": "",
            "description_identity": "",
            "description_off_topic": "",
            "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
            "clarification_questions": [
                "Could you tell me more about yourself?",
                "How would you describe your typical interactions with others?",
                "What kind of activities or work do you enjoy most?",
                "How do you typically handle challenging situations?"
            ]
        }
    except Exception as e:
        print(f"PARSING ERROR: {str(e)}")
        model_output = {
            "id": req.id,
            "status": "incomplete",
            "personal_greeting": "",
            "description_english": "",
            "description_arabic": "",
            "description_identity": "",
            "description_off_topic": "",
            "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
            "clarification_questions": [
                "Could you provide more information about yourself?",
                "How do you feel in different situations?",
                "How do you interact with others?",
                "What are your typical behaviors and habits?"
            ]
        }

    # Attach token usage if present
    if "input_tokens" in gpt_json:
        model_output["input_tokens"] = gpt_json["input_tokens"]
    if "output_tokens" in gpt_json:
        model_output["output_tokens"] = gpt_json["output_tokens"]
    if "total_tokens" in gpt_json:
        model_output["total_tokens"] = gpt_json["total_tokens"]

    # Ensure 'id' is always an integer for response validation
    try:
        model_output["id"] = int(model_output["id"])
    except Exception:
        model_output["id"] = req.id

    # Ensure all required fields exist with defaults
    model_output.setdefault("personal_greeting", "")
    model_output.setdefault("description_english", "")
    model_output.setdefault("description_arabic", "")
    model_output.setdefault("description_identity", "")
    model_output.setdefault("description_off_topic", "")
    model_output.setdefault("missing_traits", [])
    model_output.setdefault("clarification_questions", [])

    # Enforce language output: only fill in requested language(s)
    requested_langs = req.languages
    if isinstance(requested_langs, str):
        requested_langs = [requested_langs]
    requested_langs = [l.lower() for l in requested_langs]
    if not ("en" in requested_langs or "english" in requested_langs):
        model_output["description_english"] = ""
    if not ("ar" in requested_langs or "arabic" in requested_langs):
        model_output["description_arabic"] = ""

    processing_time = time.time() - t0
    print(f"Request processed in {processing_time:.2f} seconds")

    return TraitResponse(**model_output)

@app.get("/user/{user_id}/memory")
async def get_user_memory(user_id: int):
    """Get stored conversation memory for a user"""
    memory = load_user_memory(user_id)
    return {
        "user_id": user_id,
        "memory": memory,
        "timestamp": time.time()
    }

@app.delete("/user/{user_id}/memory")
async def clear_user_memory(user_id: int):
    """Clear stored conversation memory for a user"""
    if user_id in _user_memory_store:
        del _user_memory_store[user_id]
        return {"message": f"Memory cleared for user {user_id}"}
    else:
        return {"message": f"No memory found for user {user_id}"}

@app.get("/stats")
async def get_stats():
    """Get basic API usage statistics"""
    return {
        "active_users": len(_user_memory_store),
        "total_users_served": len(_user_memory_store),
        "timestamp": time.time()
    }
