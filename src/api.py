"""FastAPI service: health check and one-transaction fraud scoring."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from src.predict import load_artifact, score_transaction, InputValidationError

app = FastAPI(title="Credit Card Fraud Review API", version="1.0.0",
    description="Module 6 educational prototype; human review only.")
artifact = load_artifact()

class ScoreRequest(BaseModel):
    transaction: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": artifact is not None,
            "model_version": artifact["model_version"]}

@app.post("/v1/fraud/score")
def score(request: ScoreRequest):
    try:
        return score_transaction(request.transaction, artifact)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
