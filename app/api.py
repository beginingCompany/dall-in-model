from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
from app.predict import PersonalityPredictor
from app.GPT_api import PersonalityAnalyzer
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ValidationError
from typing import Optional, Union, List

app = FastAPI()
predictor = PersonalityPredictor(num_labels=120, top_k=3)

class PredictionRequest(BaseModel):
    text: str

class PredictionItem(BaseModel):
    class_name: str
    confidence: str

class PredictionResponse(BaseModel):
    text: str
    predictions: List[PredictionItem]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = predictor.predict([request.text]).iloc[0]
        # Print for debugging
        print(result['predictions'])
        return {
            "text": result['text'],
            "predictions": [
                {
                    "class_name": p.get("class") or p.get("class_name") or p.get("label", ""),
                    "confidence": p.get("confidence", 0.0)
                }
                for p in result['predictions']
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

analyzer = PersonalityAnalyzer()  # Instantiate once at startup

class UserRequest(BaseModel):
    id: int
    user_input: str
    new_input: Optional[str] = ""
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

@app.post("/analyze-personality", response_model=TraitResponse)
async def personality(request: Request):
    try:
        data = await request.json()
        req = UserRequest(**data)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input JSON")

    try:
        gpt_json = analyzer.analyze(req.user_input, req.new_input, req.get_languages())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Compose result
    result = {
        "id": req.id,
        "input_tokens": gpt_json.get("input_tokens"),
        "output_tokens": gpt_json.get("output_tokens"),
        "total_tokens": gpt_json.get("total_tokens"),
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
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected GPT output", "raw_response": gpt_json},
        )
    return result
