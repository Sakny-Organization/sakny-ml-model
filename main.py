from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import os

from schemas import (
    PriceEstimateRequest,
    PriceEstimateResponse,
    MatchRequest,
    MatchResponse,
    MatchScoreRequest,
    MatchScoreResponse,
    HealthResponse,
)
from matching import compute_match_score, recommend_matches
from pricing import predict_price

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = os.getenv("MODEL_PATH", "price_estimator_model.pkl")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"WARNING: Model file not found at {model_path}. Price estimation will be unavailable.")
        model = None
    yield


app = FastAPI(
    title="Sakny ML Service",
    description="Roommate matching and price estimation APIs",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.post("/api/v1/price/estimate", response_model=PriceEstimateResponse)
async def estimate_price(request: PriceEstimateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Price estimation model not loaded")
    return predict_price(model, request)


@app.post("/api/v1/match/score", response_model=MatchScoreResponse)
async def score_match(request: MatchScoreRequest):
    result = compute_match_score(request.seeker, request.candidate)
    return result


@app.post("/api/v1/match/recommend", response_model=MatchResponse)
async def recommend(request: MatchRequest):
    matches = recommend_matches(
        seeker=request.seeker,
        candidates=request.candidates,
        top_n=request.top_n,
        min_score=request.min_score,
    )
    return MatchResponse(matches=matches)
