"""
Travel AI Backend — FastAPI app.
POST /itinerary/generate and /api/itinerary/generate: accept trip params, return 3 itinerary options.
"""

import os

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    ItineraryGenerateResponse,
    PlanWithPicksRequest,
    PlanWithPicksResponse,
    QuoteRequest,
    QuoteResponse,
    TripDocumentRequest,
    TripDocumentResponse,
    TripParams,
)
from app.ollama_service import generate_itinerary
from app.plan_with_picks_service import build_plan_from_picks
from app.quote_service import build_quote
from app.trip_document_service import TripDocumentError, generate_trip_document

app = FastAPI(
    title="Travel AI Backend",
    description="Generate trip itineraries via Ollama; returns options matching the frontend JSON spec.",
    version="0.1.0",
)

_cors_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://voyager-ai.netlify.app",
]
# Optional: allow more deployed UI origins via CORS_ORIGINS env
_extra = os.environ.get("CORS_ORIGINS", "").strip()
if _extra:
    _cors_origins.extend(o.strip().rstrip("/") for o in _extra.split(",") if o.strip())
# Normalize: no trailing slash so browser Origin matches
_cors_origins = [o.rstrip("/") for o in _cors_origins]
_cors_origins_set = set(_cors_origins)


async def _cors_force_middleware(request, call_next):
    """Ensure CORS headers are on every response so preflight from Netlify/ngrok succeeds."""
    origin = request.headers.get("origin", "").rstrip("/")
    response = await call_next(request)
    if origin in _cors_origins_set:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers.setdefault("Access-Control-Allow-Credentials", "true")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, PATCH, DELETE")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization, ngrok-skip-browser-warning")
    response.headers.setdefault("Access-Control-Max-Age", "86400")
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.middleware("http")(_cors_force_middleware)


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Respond to CORS preflight for any path so Netlify + ngrok get Access-Control-* headers."""
    return {"ok": True}


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


@app.post(
    "/api/itinerary/quote",
    response_model=QuoteResponse,
    summary="In-depth quote from selected plan",
)
async def itinerary_quote_api(body: QuoteRequest):
    """Itemized breakdown, summary (subtotal + platform_fee = total), and points optimization."""
    return build_quote(body.option)


@app.post(
    "/itinerary/quote",
    response_model=QuoteResponse,
    summary="In-depth quote for selected plan (no prefix)",
)
async def itinerary_quote(body: QuoteRequest):
    """Same as /api/itinerary/quote. Request body: { \"option\": <selected option from /itinerary/generate> }."""
    return build_quote(body.option)


@app.post(
    "/api/itinerary/trip-document",
    response_model=TripDocumentResponse,
    summary="Generate full trip document (single AI call)",
)
async def trip_document_api(body: TripDocumentRequest):
    """One AI call: itinerary, suggestions, currency, mobile, card benefits, language cheat sheet. Returns { \"content\": \"...\" }."""
    try:
        return await generate_trip_document(body)
    except TripDocumentError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post(
    "/itinerary/trip-document",
    response_model=TripDocumentResponse,
    summary="Generate full trip document (no prefix)",
)
async def trip_document(body: TripDocumentRequest):
    """Same as /api/itinerary/trip-document. Frontend uses VITE_ITINERARY_API_BASE + /itinerary/trip-document."""
    try:
        return await generate_trip_document(body)
    except TripDocumentError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post(
    "/api/itinerary/plan-with-picks",
    response_model=PlanWithPicksResponse,
    summary="Build one itinerary from user picks",
)
async def plan_with_picks_api(body: PlanWithPicksRequest):
    """Build one full itinerary (daily_plan) from picks; same option shape as /itinerary/generate."""
    return build_plan_from_picks(body)


@app.post(
    "/itinerary/plan-with-picks",
    response_model=PlanWithPicksResponse,
    summary="Build one itinerary from user picks (no prefix)",
)
async def plan_with_picks(body: PlanWithPicksRequest):
    """Same as /api/itinerary/plan-with-picks. Body: picks (array of { label, google_maps_url? }), optional origin, destination, start_date, end_date."""
    return build_plan_from_picks(body)
