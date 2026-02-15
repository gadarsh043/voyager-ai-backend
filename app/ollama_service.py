"""
Call Ollama to generate itinerary options and map response to the API spec.
"""

import json
import os
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx

# Ollama timeout: allow the model enough time (up to ~1–2 minutes)
# to think and produce rich, well-structured itineraries before we fall back.
OLLAMA_TIMEOUT_SEC = 90.0

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


def get_ollama_model() -> str:
    """Model used for Ollama calls. Set env OLLAMA_MODEL to override (e.g. llama3.1:8b)."""
    return (os.environ.get("OLLAMA_MODEL") or "llama3.2").strip() or "llama3.2"


# Kept for backwards compatibility (e.g. trip_document_service)
DEFAULT_MODEL = "llama3.2"


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

    system = """You are a senior local travel planner with deep knowledge of destinations worldwide. Use YOUR TRAINING DATA and knowledge to suggest real, specific places—don't rely on generic lists.

You respond ONLY with valid JSON (no markdown, no code fences):
{
  "options": [
    {
      "id": "opt_1",
      "label": "Short title (pace + budget)",
      "daily_plan": {
        "flight_from_source": { "from_location": "XXX", "start_time": "ISO8601", "reach_by": "ISO8601" },
        "flight_to_origin": { "from_location": "XXX", "to_location": "XXX", "start_time": "ISO8601", "reach_by": "ISO8601" },
        "hotel_stay": [ { "name": "Hotel Name", "check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD", "image_url": "", "google_maps_url": "" } ],
        "days": [ { "day": 1, "activities": [ { "start_from": "...", "name": "REAL SPECIFIC PLACE NAME", "start_time": "HH:MM", "reach_time": "HH:MM", "time_to_spend": "1h 30m" } ] } ]
      },
      "total_estimated_cost": 5500
    },
    { same for "opt_2" },
    { same for "opt_3" }
  ]
}

CRITICAL – Use your knowledge:
- Draw from your training data about the destination. Use REAL place names: actual museums, markets, neighborhoods, landmarks, restaurants, parks, day-trip towns.
- For ANY destination, you know real places. Use them. Don't make up generic names.
- If the destination is less common, use your general knowledge: real neighborhoods, markets, viewpoints, cultural sites that exist there.

MANDATORY – Preferences:
- User preferences are REQUIRED. Match EVERY preference with real activities.
- "shopping" → real markets, shopping streets, department stores, boutiques (use actual names).
- "food" → real food markets, famous restaurants, food tours, street food areas.
- "culture"/"history" → real museums, historic sites, temples, galleries.
- "nature" → real parks, gardens, scenic spots, day trips to nature areas.
- "nightlife" → real evening districts, bars, shows, entertainment areas.
- Spread preferences across days; don't ignore any.

MANDATORY – No repeated places:
- NEVER repeat the same place/attraction on two different days within the same option.
- For 7 days = 21+ different places. For 4 days = 12+ different places.
- Each day introduces NEW places only.

MANDATORY – Three truly different options:
- opt_1: Different theme + different set of places (e.g. local markets, neighborhoods, budget spots).
- opt_2: Different theme + different set of places (e.g. museums, landmarks, mid-range).
- opt_3: Different theme + different set of places (e.g. day trips, premium experiences, luxury).
- Vary neighborhoods, types of activities, and actual place names so each option feels like a different trip.

Other:
- Flights/hotel: simple. Focus on places and daily flow.
- Use real place names from your knowledge. No generic labels.
- Cluster by neighborhood; include side trips (1–2 hr away) when they fit.
- Times: "HH:MM", time_to_spend like "1h 30m". Costs: clearly different (budget/mid/luxury).
"""

    prefs_note = ""
    if p.get("preferences"):
        prefs_note = f" IMPORTANT: User preferences are {p['preferences']}. You MUST include activities that match EVERY preference (e.g. shopping → markets/shops; food → food markets/restaurants; culture → museums/sites)."
    user = f"""Generate exactly 3 itinerary options for this trip. Return only the JSON object, no other text.{prefs_note}

Rules to follow:
- Every day in each option must list NEW places—never repeat an attraction on another day in that same option.
- The 3 options must offer different sets of places and themes (e.g. option 1: markets & local; option 2: museums & landmarks; option 3: day trips & premium), not the same 12 places shuffled.
- Match the user's preferences with real activities in the destination.

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


def _preference_extras(destination: str, preferences: list[str] | None) -> list[str]:
    """Extra activities to add to the pool when user has specific preferences (e.g. shopping)."""
    if not preferences:
        return []
    prefs = [str(p).lower() for p in preferences]
    d = (destination or "").lower()
    extras: list[str] = []
    if "shopping" in prefs:
        if "london" in d or "uk" in d:
            extras.extend(["Oxford Street & Regent Street", "Harrods (Knightsbridge)", "Portobello Road Market", "Liberty London", "Covent Garden market & boutiques", "Brick Lane vintage & streetwear"])
        elif "tokyo" in d or "japan" in d:
            extras.extend(["Shibuya 109 & Takeshita Street", "Ginza department stores", "Nakamise-dori (Asakusa)", "Don Quijote (Donki)", "Kappabashi kitchen street"])
        elif "paris" in d or "france" in d:
            extras.extend(["Champs-Élysées", "Galeries Lafayette", "Le Marais boutiques", "Marché aux Puces de Saint-Ouen"])
        else:
            extras.extend(["Main shopping street", "Local market for crafts & souvenirs", "Department store or mall"])
    if "food" in prefs:
        extras.extend(["Food market or food tour", "Local specialty restaurant", "Street food or casual eats"])
    if "culture" in prefs or "history" in prefs:
        extras.extend(["Museum or historic site", "Cultural quarter or heritage walk"])
    if "nature" in prefs:
        extras.extend(["Park or gardens", "Scenic viewpoint or nature spot"])
    return extras


def _generate_dynamic_activities(
    destination: str, num_days: int, preferences: list[str] | None, option_num: int
) -> list[list[str]]:
    """Generate activity names dynamically based on destination + preferences.
    This is a smarter fallback that creates varied names without hardcoded pools.
    option_num (1-3) ensures different themes per option.
    """
    d = (destination or "").strip()
    prefs = [str(p).lower() for p in (preferences or [])]
    
    # Generate base activity types based on option theme
    themes = {
        1: ["local market", "neighborhood walk", "budget restaurant", "park", "street food", "local shop", "community center", "public square"],
        2: ["museum", "landmark", "gallery", "historic site", "viewpoint", "cultural center", "monument", "cathedral"],
        3: ["day trip", "premium restaurant", "luxury experience", "scenic route", "exclusive tour", "spa", "fine dining", "private tour"],
    }
    
    base_types = themes.get(option_num, themes[2])
    
    # Add preference-specific types
    if "shopping" in prefs:
        base_types.extend(["shopping street", "market", "boutique", "department store", "craft market"])
    if "food" in prefs:
        base_types.extend(["food market", "restaurant", "food tour", "street food", "cafe", "bakery"])
    if "culture" in prefs or "history" in prefs:
        base_types.extend(["museum", "historic site", "temple", "palace", "heritage walk"])
    if "nature" in prefs:
        base_types.extend(["park", "garden", "nature reserve", "scenic spot", "hiking trail"])
    if "nightlife" in prefs:
        base_types.extend(["evening district", "bar", "show", "entertainment area"])
    
    # Generate unique names by combining destination + type + variation
    out: list[list[str]] = []
    used: set[str] = set()
    
    for day in range(num_days):
        day_activities: list[str] = []
        for slot in range(3):
            # Try different combinations until we get a unique name
            attempts = 0
            while attempts < 20:
                # Pick a type and create a specific-sounding name
                type_name = base_types[attempts % len(base_types)]
                
                # Create realistic-sounding names
                if "market" in type_name:
                    name = f"{d} {type_name}" if day == 0 else f"{d} {type_name} (day {day+1})"
                elif "day trip" in type_name:
                    name = f"Day trip from {d} to nearby town"
                elif "neighborhood" in type_name:
                    name = f"{d} neighborhood exploration"
                elif "museum" in type_name or "gallery" in type_name:
                    name = f"{d} {type_name}"
                elif "park" in type_name or "garden" in type_name:
                    name = f"{d} {type_name}"
                else:
                    name = f"{d} {type_name}"
                
                # Add variation to ensure uniqueness
                if day > 0:
                    variations = ["", " (continued)", f" - area {day+1}", f" - part {day+1}"]
                    name += variations[day % len(variations)]
                
                if name not in used:
                    used.add(name)
                    day_activities.append(name)
                    break
                attempts += 1
            else:
                # Fallback if we can't generate unique
                day_activities.append(f"{d} activity {day+1}-{slot+1}")
        
        out.append(day_activities)
    
    return out


def _destination_activities(
    destination: str, num_days: int, preferences: list[str] | None, start_index: int = 0
) -> list[list[str]]:
    """One list of 3 specific activity names per day. No place is repeated within the same option.
    start_index makes each of the 3 options get a different slice of the pool.
    
    NOTE: This uses hardcoded pools as a last resort. Prefer using Ollama's knowledge instead.
    """
    d = (destination or "").lower()
    need = num_days * 3  # at least this many unique activities per option

    # Base pools: large enough (24+) so we never repeat within one option
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
            "Yanaka Ginza (old Tokyo)",
            "Odaiba (waterfront)",
            "Daikanyama & Nakameguro",
            "Day trip: Kamakura (Great Buddha)",
            "Day trip: Nikko or Hakone",
            "Golden Gai (Shinjuku)",
            "Kiyosumi Garden",
            "Ginza stroll & art galleries",
            "Ueno Park & museums",
            "Koishikawa Korakuen",
            "Omotesando & Meiji-jingu",
            "Kappabashi (kitchen street)",
            "Don Quijote / Donki",
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
            "Perot Museum",
            "Trinity Groves",
            "AT&T Stadium (if Arlington)",
            "McKinney Avenue (Dallas)",
            "Galleria or NorthPark (shopping)",
            "Fair Park",
            "White Rock Lake",
            "Cattle drive (Fort Worth)",
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
            "Palace of Versailles (day trip)",
            "Local wine & cheese tasting",
            "Père Lachaise Cemetery",
            "Canal Saint-Martin",
            "Marché Bastille or Marché d'Aligre",
            "Saint-Germain-des-Prés",
            "Tuileries Garden",
            "Rodin Museum",
            "Galeries Lafayette",
            "Champs-Élysées",
            "Belleville or Buttes-Chaumont",
            "Île de la Cité walk",
            "Montmartre vineyards",
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
            "Oxford Street & Regent Street",
            "Harrods (Knightsbridge)",
            "Portobello Road Market",
            "Greenwich (Cutty Sark & market)",
            "Notting Hill & Portobello",
            "Brick Lane & Spitalfields",
            "Victoria and Albert Museum",
            "Kensington Palace & Gardens",
            "Leadenhall Market",
            "Shoreditch street art & cafes",
            "Liberty London",
            "Columbia Road Flower Market (Sun)",
            "Hampstead Heath",
            "Sky Garden or The Shard view",
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
            "Local market (souvenirs & clothes)",
            "Sunset point or viewpoint",
            "Street food area",
            "Palace or haveli",
            "Art or handicraft district",
            "Day trip to nearby town",
            "Evening bazaar or night market",
        ]
    else:
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
            "Day trip to nearby town",
            "Second market or bazaar",
            "Another museum or gallery",
            "Gardens or nature spot",
            "Local brewery or tasting",
            "Evening entertainment area",
        ]

    # Preference-based extras (e.g. shopping) – add to front so they get picked
    preference_extras = _preference_extras(destination, preferences)
    pool = preference_extras + [x for x in pool if x not in preference_extras]

    out: list[list[str]] = []
    used_idx = max(0, start_index)
    used_names: set[str] = set()
    for day in range(num_days):
        day_activities: list[str] = []
        for _ in range(3):
            # Pick next activity from pool that we haven't used yet in this option
            attempts = 0
            while attempts < len(pool):
                candidate = pool[used_idx % len(pool)]
                used_idx += 1
                attempts += 1
                if candidate not in used_names:
                    used_names.add(candidate)
                    day_activities.append(candidate)
                    break
            else:
                # Fallback if pool too small: reuse a generic placeholder only as last resort
                day_activities.append("Local highlight & explore")
        out.append(day_activities)
    return out


def _fallback_response(params: TripParams) -> ItineraryGenerateResponse:
    """Return a valid response when Ollama fails or is unavailable.

    Trip-aware with specific activity names, and ensures the 3 options have
    genuinely different day plans (not just different labels/costs).
    Places matter most; flights/hotels stay simple.
    """
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

    # Generate dynamic activity names for fallback (only used if Ollama fails)
    # Each option gets a different theme to ensure variety
    day_activity_names_opt1 = _generate_dynamic_activities(dest_display, num_days, params.preferences, option_num=1)
    day_activity_names_opt2 = _generate_dynamic_activities(dest_display, num_days, params.preferences, option_num=2)
    day_activity_names_opt3 = _generate_dynamic_activities(dest_display, num_days, params.preferences, option_num=3)

    def make_days(day_activity_names: list[list[str]]) -> list[DayPlan]:
        days_list: list[DayPlan] = []
        slots = [(9, 0, 11, 0, "2h"), (11, 30, 13, 0, "1h 30m"), (14, 0, 17, 0, "3h")]
        for d in range(1, num_days + 1):
            names = (
                day_activity_names[d - 1]
                if d <= len(day_activity_names)
                else ["Local highlight", "Lunch & explore", "Evening in town"]
            )
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

    def base_plan(day_activity_names: list[list[str]]) -> DailyPlan:
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
            # Keep hotel simple; user cares more about places/activities.
            hotel_stay=[
                HotelStay(
                    name="Hotel",
                    check_in=start,
                    check_out=end,
                    image_url=hotel_image,
                    google_maps_url=hotel_maps,
                ),
            ],
            days=make_days(day_activity_names),
        )

    return ItineraryGenerateResponse(
        options=[
            ItineraryOption(
                id="opt_1",
                label="Budget-friendly plan (slow & local)",
                daily_plan=base_plan(day_activity_names_opt1),
                total_estimated_cost=round(budget_val * 0.4, 0),
            ),
            ItineraryOption(
                id="opt_2",
                label="Balanced plan (mixed highlights + local)",
                daily_plan=base_plan(day_activity_names_opt2),
                total_estimated_cost=round(budget_val * 0.85, 0),
            ),
            ItineraryOption(
                id="opt_3",
                label="Luxury plan (fast-paced, more paid experiences)",
                daily_plan=base_plan(day_activity_names_opt3),
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
                    "model": get_ollama_model(),
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.8,  # Higher temperature = more creative/varied responses
                        "top_p": 0.9,  # Nucleus sampling for diversity
                        "num_predict": 4000,  # Allow longer responses for detailed itineraries
                    },
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
