"""
Call Ollama to generate itinerary options and map response to the API spec.
"""

import json
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx

# Ollama timeout: keep short so we return fallback quickly when Ollama is slow/unavailable.
OLLAMA_TIMEOUT_SEC = 10.0

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

CRITICAL - Activities must be specific and unique:
- Do NOT use generic labels like "Morning spot", "Afternoon exploration", "Explore", "Free time".
- Every activity must have a specific name: real landmarks, neighborhoods, museums, markets, or named experiences (e.g. "Senso-ji Temple", "Tsukiji Outer Market", "Shibuya Crossing", "Cooking class in Shinjuku", "Dallas Arts District", "Dealey Plaza / Sixth Floor Museum").
- Match the destination and the user's preferences (e.g. if preferences include "History", include museums or historic sites; if "Food", include a market or food tour).
- Use the "start_from" field as the previous location (e.g. "Hotel", or the previous activity's place name). Each activity should feel like a real, thought-through stop.
"""

    user = f"""Generate exactly 3 itinerary options for this trip. Return only the JSON object, no other text.

For every day, use SPECIFIC place names and activities for the destination—no generic "morning" or "afternoon" labels. Think like a local guide.

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


def _num_days(start: str, end: str) -> int:
    """Number of full days between start_date and end_date (inclusive). Min 1."""
    try:
        a = datetime.strptime(start, "%Y-%m-%d")
        b = datetime.strptime(end, "%Y-%m-%d")
        return max(1, (b - a).days + 1)
    except (ValueError, TypeError):
        return 1


def _placeholder_image_url(seed: str, width: int = 400, height: int = 300) -> str:
    """Stable placeholder image via Picsum. Same seed = same image."""
    safe = re.sub(r"[^a-z0-9]+", "", seed.lower()) or "travel"
    return f"https://picsum.photos/seed/{safe}/{width}/{height}"


def _google_maps_url(query: str) -> str:
    """Google Maps search URL for a place name + location."""
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def _fill_placeholder_images(response: ItineraryGenerateResponse, params: TripParams) -> None:
    """Ensure at least one image_url and google_maps_url in the response (mutates in place)."""
    dest = (params.destination or "destination").strip()
    hotel_img = _placeholder_image_url(f"hotel_{dest}")
    hotel_maps = _google_maps_url(f"Hotel {dest}")
    activity_img = _placeholder_image_url(f"activity_{dest}")
    activity_maps = _google_maps_url(f"Attractions {dest}")
    filled_activity = False
    for opt in response.options:
        plan = opt.daily_plan
        if plan.hotel_stay:
            for stay in plan.hotel_stay:
                if not stay.image_url:
                    stay.image_url = hotel_img
                if not stay.google_maps_url:
                    stay.google_maps_url = hotel_maps
        for day in plan.days:
            for act in day.activities:
                if not filled_activity and (not act.image_url or not act.google_maps_url):
                    if not act.image_url:
                        act.image_url = activity_img
                    if not act.google_maps_url:
                        act.google_maps_url = activity_maps
                    filled_activity = True
                    break
            if filled_activity:
                break
        if filled_activity:
            break


def _destination_activities(destination: str, num_days: int, preferences: list[str] | None) -> list[list[str]]:
    """Return one list of 3 specific activity names per day, destination-aware and unique (no generic 'morning/afternoon')."""
    d = (destination or "").lower()

    # Pool of specific activities per region; we pick 3 per day in order
    if "tokyo" in d or "japan" in d or "osaka" in d or "kyoto" in d:
        pool = [
            "Senso-ji Temple (Asakusa)",
            "Tsukiji Outer Market",
            "Shibuya Crossing & Hachiko",
            "teamLab Borderless",
            "Imperial Palace East Gardens",
            "Akihabara district",
            "Meiji Shrine & Yoyogi Park",
            "Shinjuku Gyoen",
            "Roppongi Hills / Mori Art Museum",
            "Tokyo Skytree",
            "Ramen museum or local ramen spot",
            "Traditional tea ceremony experience",
        ]
    elif "dallas" in d or "houston" in d or "texas" in d:
        pool = [
            "Dallas Arts District",
            "Dealey Plaza & Sixth Floor Museum",
            "Fort Worth Stockyards",
            "Klyde Warren Park",
            "Dallas Arboretum" if "dallas" in d else "Houston Museum District",
            "Reunion Tower" if "dallas" in d else "Space Center Houston",
            "Deep Ellum (food & murals)",
            "Bishop Arts District",
            "George W. Bush Presidential Library" if "dallas" in d else "Buffalo Bayou Park",
            "Local BBQ or Tex-Mex spot",
            "Nasher Sculpture Center" if "dallas" in d else "Menil Collection",
            "Neighborhood food tour",
        ]
    elif "paris" in d or "france" in d:
        pool = [
            "Louvre Museum",
            "Eiffel Tower & Champ de Mars",
            "Notre-Dame area & Île de la Cité",
            "Montmartre & Sacré-Cœur",
            "Musée d'Orsay",
            "Seine river walk & Pont Alexandre III",
            "Le Marais (markets & falafel)",
            "Latin Quarter & Panthéon",
            "Sainte-Chapelle",
            "Café culture & pastry stop",
            "Palace of Versailles (if day trip)",
            "Local wine & cheese tasting",
        ]
    elif "london" in d or "uk" in d:
        pool = [
            "British Museum",
            "Tower of London & Tower Bridge",
            "Westminster & Big Ben",
            "Hyde Park & Kensington Gardens",
            "Camden Market",
            "Natural History Museum",
            "South Bank & Tate Modern",
            "Borough Market",
            "Covent Garden",
            "St Paul's Cathedral",
            "Afternoon tea experience",
            "West End show or walk",
        ]
    elif "india" in d or "hyderabad" in d or "mumbai" in d or "delhi" in d:
        pool = [
            "Charminar & Laad Bazaar" if "hyderabad" in d else "Historic fort or monument",
            "Golconda Fort" if "hyderabad" in d else "Old city walk",
            "Ramoji Film City" if "hyderabad" in d else "Local market & street food",
            "Salar Jung Museum" if "hyderabad" in d else "Museum or palace",
            "Hussain Sagar Lake",
            "Birla Mandir",
            "Traditional biryani or chai stop",
            "Heritage walk",
            "Craft or bazaar shopping",
            "Temple or cultural site",
            "Food tour (local specialties)",
            "Gardens or park",
        ]
    else:
        # Generic but still specific-sounding
        pool = [
            "Historic center / main square",
            "Local market or food hall",
            "Top museum or gallery",
            "Famous landmark or viewpoint",
            "Waterfront or park",
            "Neighborhood walk & coffee",
            "Local specialty food stop",
            "Cultural or heritage site",
            "Shopping street or district",
            "Sunset or evening stroll",
            "Guided tour (history or food)",
            "Hidden gem recommended by locals",
        ]

    out: list[list[str]] = []
    used = 0
    for day in range(num_days):
        day_activities: list[str] = []
        for _ in range(3):
            day_activities.append(pool[used % len(pool)])
            used += 1
        out.append(day_activities)
    return out


def _fallback_response(params: TripParams) -> ItineraryGenerateResponse:
    """Return a valid response when Ollama fails or is unavailable. Trip-aware with specific activity names."""
    def code(s: str | None, default: str) -> str:
        if not s:
            return default
        s = s.strip().upper()
        for part in s.replace(",", " ").split():
            if len(part) >= 2:
                return part[:3]
        return default[:3]

    origin = code(params.origin, "ORIG")
    dest = code(params.destination, "DEST")
    dest_display = (params.destination or "destination").strip()
    start = params.start_date or "2026-03-15"
    end = params.end_date or "2026-03-19"
    num_days = _num_days(start, end)

    hotel_image = _placeholder_image_url(f"hotel_{dest_display}")
    hotel_maps = _google_maps_url(f"Hotel {dest_display}")
    activity_image = _placeholder_image_url(f"activity_{dest_display}")
    activity_maps = _google_maps_url(f"Attractions {dest_display}")

    budget_val: float = 3000.0
    if params.budget is not None:
        try:
            budget_val = float(params.budget)
        except (TypeError, ValueError):
            pass

    day_activity_names = _destination_activities(dest_display, num_days, params.preferences)

    def make_days() -> list[DayPlan]:
        days_list: list[DayPlan] = []
        slots = [(9, 0, 11, 0, "2h"), (11, 30, 13, 0, "1h 30m"), (14, 0, 17, 0, "3h")]
        for d in range(1, num_days + 1):
            names = day_activity_names[d - 1] if d <= len(day_activity_names) else ["Local highlight", "Lunch & explore", "Evening in town"]
            prev = "Hotel" if d > 1 else "Airport"
            activities: list[Activity] = []
            for i, name in enumerate(names):
                if i >= len(slots):
                    break
                sh, sm, rh, rm, dur = slots[i]
                act = Activity(
                    start_from=prev,
                    start_time=f"{sh:02d}:{sm:02d}",
                    reach_time=f"{rh:02d}:{rm:02d}",
                    time_to_spend=dur,
                    name=name,
                    image_url=activity_image if d == 1 and i == 0 else None,
                    google_maps_url=activity_maps if d == 1 and i == 0 else None,
                )
                activities.append(act)
                prev = name
            days_list.append(DayPlan(day=d, activities=activities))
        return days_list

    def base_plan() -> DailyPlan:
        return DailyPlan(
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
                    name="Hotel",
                    check_in=start,
                    check_out=end,
                    image_url=hotel_image,
                    google_maps_url=hotel_maps,
                ),
            ],
            days=make_days(),
        )

    return ItineraryGenerateResponse(
        options=[
            ItineraryOption(
                id="opt_1",
                label="Budget-friendly plan",
                daily_plan=base_plan(),
                total_estimated_cost=round(budget_val * 0.4, 0),
            ),
            ItineraryOption(
                id="opt_2",
                label="Balanced plan",
                daily_plan=base_plan(),
                total_estimated_cost=round(budget_val * 0.85, 0),
            ),
            ItineraryOption(
                id="opt_3",
                label="Luxury plan",
                daily_plan=base_plan(),
                total_estimated_cost=round(budget_val * 1.5, 0),
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
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SEC) as client:
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
    _fill_placeholder_images(mapped, params)
    return mapped
