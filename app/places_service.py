"""
Google Places service: fetch real tourist attractions and restaurants for a destination.
Used to pre-populate the Ollama itinerary prompt with actual place names so the LLM
only needs to *schedule* — not recall — places.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

# Fetch enough places so we can give each of the 3 options its own non-overlapping pool.
# 3 options × 4 days × 3 activities = 36 minimum. We fetch 90+ for comfortable headroom.
MAX_PLACES = 90


@dataclass
class PlaceInfo:
    name: str
    google_maps_url: str
    types: list[str] = field(default_factory=list)
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    description: Optional[str] = None  # editorial_summary or vicinity
    price_level: Optional[int] = None
    image_url: Optional[str] = None

def _maps_url(place_id: str) -> str:
    return f"https://www.google.com/maps/place/?q=place_id:{place_id}"


def _get_api_key() -> str:
    return os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()


# Maps Google place types → friendly human labels for the description
_TYPE_LABELS: dict[str, str] = {
    "tourist_attraction": "Popular tourist attraction",
    "museum": "Museum",
    "park": "Park & green space",
    "amusement_park": "Amusement & theme park",
    "art_gallery": "Art gallery",
    "restaurant": "Restaurant",
    "cafe": "Café",
    "bar": "Bar & nightlife",
    "night_club": "Nightclub",
    "shopping_mall": "Shopping mall",
    "store": "Shopping & retail",
    "lodging": "Hotel & accommodation",
    "spa": "Spa & wellness",
    "stadium": "Stadium & sports venue",
    "aquarium": "Aquarium",
    "zoo": "Zoo & wildlife",
    "movie_theater": "Cinema",
    "library": "Library",
    "church": "Church & place of worship",
    "hindu_temple": "Temple",
    "mosque": "Mosque",
    "synagogue": "Synagogue",
    "natural_feature": "Natural landmark",
    "campground": "Campground & outdoor recreation",
    "rv_park": "RV park",
    "cemetery": "Historic cemetery",
    "embassy": "Embassy",
    "local_government_office": "Government & civic building",
    "university": "University & campus",
}


def _type_description(types: list[str]) -> str | None:
    """Generate a short, human-readable description from Google Places type tags."""
    labels = []
    for t in types:
        label = _TYPE_LABELS.get(t)
        if label and label not in labels:
            labels.append(label)
        if len(labels) == 2:
            break
    return " · ".join(labels) if labels else None


async def _text_search(
    client: httpx.AsyncClient,
    query: str,
    api_key: str,
    place_type: Optional[str] = None,
    page_token: Optional[str] = None,
) -> list[PlaceInfo]:
    """Run one Places Text Search and return PlaceInfo list."""
    params: dict = {"query": query, "key": api_key}
    if place_type:
        params["type"] = place_type
    if page_token:
        params["pagetoken"] = page_token

    try:
        r = await client.get(PLACES_TEXT_SEARCH_URL, params=params, timeout=12.0)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("Places text search failed for %r: %s", query, exc)
        return []

    results = data.get("results") or []
    places: list[PlaceInfo] = []
    for item in results:
        place_id = item.get("place_id")
        name = item.get("name", "").strip()
        if not name or not place_id:
            continue
        types = item.get("types") or []
        rating = item.get("rating")
        user_ratings_total = item.get("user_ratings_total")
        price_level = item.get("price_level")
        
        # Grab the first photo
        photos = item.get("photos", [])
        image_url = None
        if photos and isinstance(photos, list) and len(photos) > 0:
            photo_ref = photos[0].get("photo_reference")
            if photo_ref:
                image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photo_reference={photo_ref}&key={api_key}"

        # editorial_summary (Details-only) and vicinity rarely populate in Text Search;
        # fall back to a human-readable label derived from the place's type tags
        editorial = (item.get("editorial_summary") or {}).get("overview", "")
        vicinity = item.get("vicinity", "")
        description = (editorial or vicinity or "").strip() or _type_description(types)
        places.append(
            PlaceInfo(
                name=name,
                google_maps_url=_maps_url(place_id),
                types=types,
                rating=rating,
                user_ratings_total=user_ratings_total,
                description=description,
                price_level=price_level,
                image_url=image_url,
            )
        )
    return places


def _build_queries_categorized(destination: str, preferences: list[str] | None) -> dict[str, list[tuple[str, Optional[str]]]]:
    """
    Build structured queries mapped to exactly 3 categories (go, eat, stay).
    Limits redundant calls but gets enough distinct places to feed Ollama schemas.
    """
    d = destination.strip()
    prefs = [p.lower() for p in (preferences or [])]

    # Where to Go
    go_queries: list[tuple[str, Optional[str]]] = [
        (f"top tourist attractions in {d}", "tourist_attraction"),
        (f"famous landmarks and historic sites in {d}", None),
        (f"hidden gems and parks in {d}", "park"),
        (f"museums and art galleries in {d}", "museum"),
        (f"things to do and experiences in {d}", None)
    ]
    if "shopping" in prefs:
        go_queries.append((f"best shopping markets and malls in {d}", "shopping_mall"))
    if "nightlife" in prefs:
        go_queries.append((f"nightlife districts bars clubs in {d}", "night_club"))
    if "nature" in prefs:
        go_queries.append((f"nature hikes scenic spots near {d}", None))

    # Where to Eat
    eat_queries: list[tuple[str, Optional[str]]] = [
        (f"best restaurants in {d}", "restaurant"),
        (f"local food markets and street food in {d}", "restaurant"),
        (f"top cafes and bakeries in {d}", "cafe"),
        (f"casual lunch spots in {d}", "restaurant"),
    ]
    if "food" in prefs:
        eat_queries.append((f"michelin star or fine dining in {d}", "restaurant"))

    # Where to Stay
    stay_queries: list[tuple[str, Optional[str]]] = [
        (f"top rated hotels in {d}", "lodging"),
        (f"best places to stay in {d}", "lodging"),
        (f"boutique hotels and resorts in {d}", "lodging")
    ]

    return {"go": go_queries, "eat": eat_queries, "stay": stay_queries}

def _deduplicate(places: list[PlaceInfo]) -> list[PlaceInfo]:
    """Deduplicate by lowercased name; keep highest-rated duplicate."""
    seen: dict[str, PlaceInfo] = {}
    for p in places:
        key = p.name.lower().strip()
        if key not in seen:
            seen[key] = p
        else:
            existing_rating = seen[key].rating or 0
            new_rating = p.rating or 0
            if new_rating > existing_rating:
                seen[key] = p
    return list(seen.values())


def _sort_by_relevance(places: list[PlaceInfo]) -> list[PlaceInfo]:
    """Sort by type priority first, then rating."""
    priority_types = {"tourist_attraction", "museum", "restaurant", "park", "amusement_park"}

    def score(p: PlaceInfo) -> tuple[int, float]:
        type_score = 1 if any(t in priority_types for t in p.types) else 0
        return (type_score, p.rating or 0.0)

    return sorted(places, key=score, reverse=True)


def partition_places(places: list[PlaceInfo]) -> tuple[list[PlaceInfo], list[PlaceInfo], list[PlaceInfo]]:
    """
    Split the place list into 3 non-overlapping groups for the 3 itinerary options.
    Interleave instead of slicing to ensure each group gets a mix of high- and lower-rated places.
    """
    g1, g2, g3 = [], [], []
    for i, p in enumerate(places):
        if i % 3 == 0:
            g1.append(p)
        elif i % 3 == 1:
            g2.append(p)
        else:
            g3.append(p)
    return g1, g2, g3


async def fetch_destination_places(
    destination: str,
    preferences: list[str] | None = None,
) -> dict[str, list[PlaceInfo]]:
    """
    Fetch real places for a destination using Google Places Text Search.
    Returns categorized dictionary: "go", "eat", and "stay".
    """
    empty_dict = {"go": [], "eat": [], "stay": []}
    api_key = _get_api_key()
    if not api_key:
        logger.info("GOOGLE_MAPS_API_KEY not set; skipping Places pre-fetch")
        return empty_dict

    if not destination or not destination.strip():
        return empty_dict

    queries_by_cat = _build_queries_categorized(destination.strip(), preferences)
    results_by_cat: dict[str, list[PlaceInfo]] = empty_dict.copy()

    async with httpx.AsyncClient() as client:
        for cat, queries in queries_by_cat.items():
            tasks = [_text_search(client, q, api_key, ptype) for q, ptype in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_places: list[PlaceInfo] = []
            for r in results:
                if isinstance(r, list):
                    all_places.extend(r)
            
            unique = _deduplicate(all_places)
            ranked = _sort_by_relevance(unique)
            # Cap top per category (50 for go, 30 for eat, 15 for stay)
            cap = 50 if cat == "go" else (30 if cat == "eat" else 15)
            results_by_cat[cat] = ranked[:cap]
            logger.info("Places API: %d unique places fetched for %r in category %r", len(results_by_cat[cat]), destination, cat)

    return results_by_cat
