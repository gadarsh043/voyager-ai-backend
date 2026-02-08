"""
Call Ollama to generate itinerary options and map response to the API spec.
"""

import json
import re
from typing import Any

import httpx

from app.models import (
    Activity,
    DailyPlan,
    DayPlan,
    FlightLeg,
    HotelStay,
    ItineraryGenerateResponse,
    ItineraryOption,
    TripParams,
)

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_CHAT = f"{OLLAMA_BASE}/api/chat"
DEFAULT_MODEL = "llama3.2"  # or "mistral", "gemma2", etc.


def _default_trip_params() -> TripParams:
    return TripParams(
        origin="SFO",
        destination="Tokyo",
        start_date="2026-03-15",
        end_date="2026-03-19",
        budget="moderate",
        preferences=["culture", "food", "sightseeing"],
    )


def _build_prompt(params: TripParams) -> tuple[str, str]:
    p = params.model_dump(exclude_none=True)
    if not p:
        params = _default_trip_params()
        p = params.model_dump(exclude_none=True)

    system = """You are a travel itinerary assistant. You respond ONLY with valid JSON.
Your response must be a single JSON object with this exact structure (no markdown, no code fences):
{
  "options": [
    {
      "id": "opt_1",
      "label": "Short title for this plan",
      "daily_plan": {
        "flight_from_source": { "from_location": "XXX", "start_time": "ISO8601", "reach_by": "ISO8601" },
        "flight_to_origin": { "from_location": "XXX", "to_location": "XXX", "start_time": "ISO8601", "reach_by": "ISO8601" },
        "hotel_stay": [ { "name": "Hotel Name", "check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD", "image_url": "", "google_maps_url": "" } ],
        "days": [ { "day": 1, "activities": [ { "start_from": "...", "start_time": "HH:MM", "reach_time": "HH:MM", "time_to_spend": "1h 30m", "image_url": "", "google_maps_url": "" } ] } ]
      },
      "total_estimated_cost": 5500
    },
    { same for "opt_2" },
    { same for "opt_3" }
  ]
}
Rules:
- Always return exactly 3 options (opt_1, opt_2, opt_3). Vary them (e.g. budget / balanced / luxury, or different day counts).
- Use real-looking locations, hotels, and activities. Times and dates must be consistent with the trip dates.
- Flights: use ISO 8601 for start_time and reach_by. For flight_to_origin include to_location.
- Activities: start_time/reach_time can be "09:00"/"09:45". time_to_spend like "1h 30m".
- total_estimated_cost is a number (e.g. dollars).
- You may omit optional fields (image_url, google_maps_url) or set to empty string.
"""

    user = f"""Generate exactly 3 itinerary options for this trip. Return only the JSON object, no other text.

Trip parameters: {json.dumps(p, default=str)}"""

    return system, user


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON from model output, stripping markdown code blocks if present."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _map_to_response(raw: dict[str, Any]) -> ItineraryGenerateResponse | None:
    """Map parsed JSON to Pydantic response model. Returns None if invalid."""
    try:
        options = raw.get("options")
        if not isinstance(options, list) or len(options) < 1:
            return None
        # Ensure exactly 3 options with fixed ids
        out = []
        for i, opt in enumerate(options[:3]):
            if not isinstance(opt, dict):
                continue
            opt_id = opt.get("id") or f"opt_{i + 1}"
            opt["id"] = opt_id
            out.append(ItineraryOption.model_validate(opt))
        if not out:
            return None
        return ItineraryGenerateResponse(options=out)
    except Exception:
        return None


def _fallback_response(params: TripParams) -> ItineraryGenerateResponse:
    """Return a valid minimal response when Ollama fails or is unavailable."""
    origin = (params.origin or "SFO").upper()[:3]
    dest = (params.destination or "Tokyo").upper()[:3]
    start = params.start_date or "2026-03-15"
    end = params.end_date or "2026-03-19"

    return ItineraryGenerateResponse(
        options=[
            ItineraryOption(
                id="opt_1",
                label="Budget-friendly plan",
                daily_plan=DailyPlan(
                    flight_from_source=FlightLeg(
                        from_location=origin,
                        start_time=f"{start}T08:00:00Z",
                        reach_by=f"{start}T18:00:00Z",
                    ),
                    flight_to_origin=FlightLeg(
                        from_location=dest,
                        to_location=origin,
                        start_time=f"{end}T10:00:00Z",
                        reach_by=f"{end}T20:00:00Z",
                    ),
                    hotel_stay=[
                        HotelStay(
                            name="Central Budget Hotel",
                            check_in=start,
                            check_out=end,
                        )
                    ],
                    days=[
                        DayPlan(
                            day=1,
                            activities=[
                                Activity(
                                    start_from="Hotel",
                                    start_time="09:00",
                                    reach_time="09:30",
                                    time_to_spend="2h",
                                )
                            ],
                        )
                    ],
                ),
                total_estimated_cost=1200.0,
            ),
            ItineraryOption(
                id="opt_2",
                label="Balanced plan",
                daily_plan=DailyPlan(
                    flight_from_source=FlightLeg(
                        from_location=origin,
                        start_time=f"{start}T08:00:00Z",
                        reach_by=f"{start}T18:00:00Z",
                    ),
                    flight_to_origin=FlightLeg(
                        from_location=dest,
                        to_location=origin,
                        start_time=f"{end}T10:00:00Z",
                        reach_by=f"{end}T20:00:00Z",
                    ),
                    hotel_stay=[
                        HotelStay(
                            name="Mid-range City Hotel",
                            check_in=start,
                            check_out=end,
                        )
                    ],
                    days=[
                        DayPlan(
                            day=1,
                            activities=[
                                Activity(
                                    start_from="Hotel",
                                    start_time="09:00",
                                    reach_time="12:00",
                                    time_to_spend="3h",
                                )
                            ],
                        )
                    ],
                ),
                total_estimated_cost=2800.0,
            ),
            ItineraryOption(
                id="opt_3",
                label="Luxury plan",
                daily_plan=DailyPlan(
                    flight_from_source=FlightLeg(
                        from_location=origin,
                        start_time=f"{start}T08:00:00Z",
                        reach_by=f"{start}T18:00:00Z",
                    ),
                    flight_to_origin=FlightLeg(
                        from_location=dest,
                        to_location=origin,
                        start_time=f"{end}T10:00:00Z",
                        reach_by=f"{end}T20:00:00Z",
                    ),
                    hotel_stay=[
                        HotelStay(
                            name="Luxury Resort",
                            check_in=start,
                            check_out=end,
                        )
                    ],
                    days=[
                        DayPlan(
                            day=1,
                            activities=[
                                Activity(
                                    start_from="Hotel",
                                    start_time="09:00",
                                    reach_time="17:00",
                                    time_to_spend="Full day",
                                )
                            ],
                        )
                    ],
                ),
                total_estimated_cost=5500.0,
            ),
        ]
    )


async def generate_itinerary(params: TripParams) -> ItineraryGenerateResponse:
    """
    Call Ollama to generate 3 itinerary options, then validate and return the response.
    Falls back to a minimal valid response if Ollama is unavailable or returns invalid JSON.
    """
    if not any(params.model_dump().values()):
        params = _default_trip_params()

    system, user = _build_prompt(params)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                OLLAMA_CHAT,
                json={
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "format": "json",
                },
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, httpx.ConnectError, Exception):
        return _fallback_response(params)

    content = (data.get("message") or {}).get("content") or ""
    if isinstance(content, dict):
        content = json.dumps(content)
    raw = _extract_json(content)
    if not raw:
        # Try parsing content as-is (Ollama may return raw JSON when format=json)
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            return _fallback_response(params)

    mapped = _map_to_response(raw)
    if mapped is None:
        return _fallback_response(params)
    return mapped
