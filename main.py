"""
Travel AI Backend — FastAPI app.
Itinerary generation, flight tracking, destinations, quote, trip document.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.destinations_service import get_destinations
from app.flight_service import search_flights, track_flight
from app.models import (
    DestinationsResponse,
    FlightSearchRequest,
    FlightSearchResponse,
    FlightTrackRequest,
    FlightTrackResponse,
    ItineraryGenerateResponse,
    PlanWithPicksRequest,
    PlanWithPicksResponse,
    QuoteRequest,
    QuoteResponse,
    TripDocumentRequest,
    TripDocumentResponse,
    TripParams,
)
from app.ollama_service import ItineraryAPIError, generate_itinerary
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
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
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


# ----- Flight tracking -----


@app.post(
    "/api/flights/track",
    response_model=FlightTrackResponse,
    summary="Track flight by number",
)
async def flights_track_api(body: FlightTrackRequest):
    """Track a flight by airline code + number (e.g. UA1234). Returns status: scheduled, active, landed, cancelled, or invalid/unavailable/not_found."""
    return track_flight(body.flight_number)


@app.post(
    "/flights/track",
    response_model=FlightTrackResponse,
    summary="Track flight (no prefix)",
)
async def flights_track(body: FlightTrackRequest):
    """Same as /api/flights/track. Body: { \"flight_number\": \"UA1234\" }."""
    return track_flight(body.flight_number)


@app.post(
    "/api/flights/search",
    response_model=FlightSearchResponse,
    summary="Search flights for route and date",
)
async def flights_search_api(body: FlightSearchRequest):
    """Search real flights by origin/destination (city names) and date. Uses Aviation Stack dep_iata/arr_iata."""
    return search_flights(origin=body.origin, destination=body.destination, date=body.date)


@app.post(
    "/flights/search",
    response_model=FlightSearchResponse,
    summary="Search flights (no prefix)",
)
async def flights_search(body: FlightSearchRequest):
    """Same as /api/flights/search. Body: { \"origin\": \"Houston\", \"destination\": \"Tokyo\", \"date\": \"2026-03-15\" }."""
    return search_flights(origin=body.origin, destination=body.destination, date=body.date)


# ----- Destinations (inspiration) -----


@app.get(
    "/api/destinations",
    response_model=DestinationsResponse,
    summary="List curated destinations",
)
async def destinations_api():
    """Curated destinations for inspiration. Optional: seed from Hugging Face DeepNLP/travel-ai-agent via DESTINATIONS_USE_HF=1."""
    return get_destinations(trending=False)


@app.get(
    "/destinations",
    response_model=DestinationsResponse,
    summary="List destinations (no prefix)",
)
async def destinations():
    return get_destinations(trending=False)


@app.get(
    "/api/destinations/trending",
    response_model=DestinationsResponse,
    summary="Trending destinations",
)
async def destinations_trending_api():
    return get_destinations(trending=True)


@app.get(
    "/destinations/trending",
    response_model=DestinationsResponse,
    summary="Trending destinations (no prefix)",
)
async def destinations_trending():
    return get_destinations(trending=True)


async def _generate(body: TripParams | None):
    params = body or TripParams()
    try:
        return await generate_itinerary(params)
    except ItineraryAPIError as e:
        raise HTTPException(status_code=503, detail=e.message)


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
    """Same as /api/itinerary/generate. Returns 503 with detail message if Ollama fails (no generic fallback)."""
    return await _generate(body)


@app.post(
    "/api/itinerary/quote",
    response_model=QuoteResponse,
    summary="In-depth quote from selected plan",
)
async def itinerary_quote_api(body: QuoteRequest):
    """Itemized breakdown, summary (subtotal + platform_fee = total), and points optimization. Optional: num_persons for per_person."""
    return build_quote(body.option, num_persons=body.num_persons)


@app.post(
    "/itinerary/quote",
    response_model=QuoteResponse,
    summary="In-depth quote for selected plan (no prefix)",
)
async def itinerary_quote(body: QuoteRequest):
    """Same as /api/itinerary/quote. Body: { \"option\": <option>, \"num_persons\": optional }."""
    return build_quote(body.option, num_persons=body.num_persons)


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
