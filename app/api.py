from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
from app.predict import PersonalityPredictor

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
        return {
            "text": result['text'],
            "predictions": [
                {"class_name": p["class"], "confidence": p["confidence"]}
                for p in result['predictions']
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)