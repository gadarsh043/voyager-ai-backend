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
            )
        )
    return places


def _build_queries(destination: str, preferences: list[str] | None) -> list[tuple[str, Optional[str]]]:
    """
    Return (query, optional_place_type) tuples to run in parallel.
    Run many varied queries to ensure we get 90+ distinct places.
    """
    d = destination.strip()
    queries: list[tuple[str, Optional[str]]] = [
        # Core tourist & landmark
        (f"top tourist attractions in {d}", "tourist_attraction"),
        (f"famous landmarks in {d}", "tourist_attraction"),
        (f"hidden gems and local spots in {d}", "tourist_attraction"),
        (f"museums and galleries in {d}", "museum"),
        (f"parks and gardens in {d}", "park"),
        (f"historic sites in {d}", None),
        # Food & drink (get lots of variety)
        (f"best restaurants in {d}", "restaurant"),
        (f"local food markets and street food in {d}", "restaurant"),
        (f"cafes and coffee shops in {d}", "cafe"),
        (f"rooftop bars and lounges in {d}", "bar"),
        # Entertainment & experience
        (f"things to do in {d}", None),
        (f"entertainment and activities in {d}", None),
        (f"viewpoints and observation decks in {d}", None),
        # Broader radius — nearby day trips
        (f"day trips from {d} within 100 miles", "tourist_attraction"),
        (f"nearby attractions 100 miles from {d}", None),
    ]

    prefs = [p.lower() for p in (preferences or [])]
    if "shopping" in prefs:
        queries.append((f"best shopping markets and malls in {d}", "shopping_mall"))
        queries.append((f"vintage markets and boutiques in {d}", None))
    if "nightlife" in prefs:
        queries.append((f"nightlife districts bars clubs in {d}", "night_club"))
    if "nature" in prefs:
        queries.append((f"nature hikes scenic spots in {d}", None))
        queries.append((f"beaches or lakes near {d}", None))
    if "history" in prefs or "culture" in prefs:
        queries.append((f"historical heritage neighborhoods in {d}", None))
        queries.append((f"cultural centers and art districts in {d}", None))
    if "food" in prefs:
        queries.append((f"famous food tours and cooking classes in {d}", None))
        queries.append((f"michelin star restaurants near {d}", "restaurant"))

    return queries


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
) -> list[PlaceInfo]:
    """
    Fetch real places for a destination using Google Places Text Search.
    Returns up to MAX_PLACES deduplicated, sorted places.
    Falls back to empty list on any error (Ollama will generate without hints).
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info("GOOGLE_MAPS_API_KEY not set; skipping Places pre-fetch")
        return []

    if not destination or not destination.strip():
        return []

    queries = _build_queries(destination.strip(), preferences)

    async with httpx.AsyncClient() as client:
        tasks = [_text_search(client, q, api_key, ptype) for q, ptype in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_places: list[PlaceInfo] = []
    for r in results:
        if isinstance(r, list):
            all_places.extend(r)

    unique = _deduplicate(all_places)
    ranked = _sort_by_relevance(unique)
    total = ranked[:MAX_PLACES]
    logger.info("Places API: %d unique places fetched for %r", len(total), destination)
    return total
