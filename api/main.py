from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.recommender import SHLRecommender
import os

app = FastAPI()

# Load the recommender
data_path = os.getenv("DATA_PATH", "../data/assessments_structured.json")
recommender = SHLRecommender(data_path=data_path)

class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 3

class RecommendationResponse(BaseModel):
    Name: str
    Description: str
    Duration: str
    TestTypes: list
    URL: str
    similarity: float
    name_match: float
    final_score: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommendations", response_model=list[RecommendationResponse])
def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = recommender.recommend(request.query, top_k=request.top_k)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))