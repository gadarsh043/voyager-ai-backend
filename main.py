"""
Travel AI Backend — FastAPI app.
POST /itinerary/generate and /api/itinerary/generate: accept trip params, return 3 itinerary options.
"""

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import ItineraryGenerateResponse, TripParams
from app.ollama_service import generate_itinerary

app = FastAPI(
    title="Travel AI Backend",
    description="Generate trip itineraries via Ollama; returns options matching the frontend JSON spec.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "Travel AI Backend", "docs": "/docs"}


async def _generate(body: TripParams | None):
    params = body or TripParams()
    return await generate_itinerary(params)


@app.post(
    "/api/itinerary/generate",
    response_model=ItineraryGenerateResponse,
    summary="Generate itinerary options",
)
async def itinerary_generate_api(body: TripParams | None = Body(default=None)):
    return await _generate(body)


@app.post(
    "/itinerary/generate",
    response_model=ItineraryGenerateResponse,
    summary="Generate itinerary options (no prefix)",
)
async def itinerary_generate(body: TripParams | None = Body(default=None)):
    """Same as /api/itinerary/generate; supports frontend calling /itinerary/generate."""
    return await _generate(body)
