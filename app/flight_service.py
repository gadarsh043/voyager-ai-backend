"""
Flight tracking via Aviation Stack API.
Requires AVIATION_STACK_API_KEY in environment (free tier: 100 req/month).
"""

import os
import re
import time

import httpx

from app.models import (
    FlightSearchArrival,
    FlightSearchDeparture,
    FlightSearchItem,
    FlightSearchResponse,
    FlightTrackArrival,
    FlightTrackDeparture,
    FlightTrackResponse,
)

AVIATION_STACK_BASE = "https://api.aviationstack.com/v1"
TIMEOUT_SEC = 10.0

# City name (normalized) → primary IATA code for flight search
_CITY_TO_IATA: dict[str, str] = {
    "houston": "IAH",
    "tokyo": "NRT",
    "london": "LHR",
    "paris": "CDG",
    "new york": "JFK",
    "newyork": "JFK",
    "san francisco": "SFO",
    "sanfrancisco": "SFO",
    "los angeles": "LAX",
    "losangeles": "LAX",
    "chicago": "ORD",
    "dallas": "DFW",
    "miami": "MIA",
    "mumbai": "BOM",
    "delhi": "DEL",
    "bangalore": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "singapore": "SIN",
    "hong kong": "HKG",
    "hongkong": "HKG",
    "dubai": "DXB",
    "barcelona": "BCN",
    "rome": "FCO",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "sydney": "SYD",
    "melbourne": "MEL",
    "seoul": "ICN",
    "beijing": "PEK",
    "shanghai": "PVG",
    "boston": "BOS",
    "washington": "IAD",
    "seattle": "SEA",
    "denver": "DEN",
    "atlanta": "ATL",
    "philadelphia": "PHL",
    "phoenix": "PHX",
    "las vegas": "LAS",
    "orlando": "MCO",
    "narita": "NRT",
    "haneda": "HND",
    "iah": "IAH",
    "nrt": "NRT",
    "hnd": "HND",
}


def _get_api_key() -> str | None:
    return os.environ.get("AVIATION_STACK_API_KEY") or os.environ.get("AVIATIONSTACK_API_KEY")


def _parse_flight_number(flight_number: str) -> str | None:
    """Normalize to IATA flight code (e.g. UA1234, AA456). Returns None if invalid."""
    s = (flight_number or "").strip().upper()
    if not s:
        return None
    # Allow "UA 1234", "UA1234", "AA456"
    s = re.sub(r"\s+", "", s)
    # Must start with 2 letters and have at least 1 digit
    if re.match(r"^[A-Z0-9]{2,}[0-9]+$", s):
        return s
    return None


def _format_airport(airport_name: str | None, iata: str | None) -> str:
    """e.g. 'SFO - San Francisco International'."""
    if not airport_name:
        return (iata or "Unknown").strip()
    if iata:
        return f"{iata} - {airport_name.strip()}"
    return airport_name.strip()


def _normalize_status(api_status: str | None) -> str:
    """Map API status to spec: scheduled, active, landed, cancelled, etc."""
    if not api_status:
        return "unknown"
    s = (api_status or "").lower()
    if s in ("scheduled", "active", "landed", "cancelled", "incident", "diverted"):
        return s
    return "scheduled"


# Cache for flight search: (origin, dest, date) -> (response, expiry_time). Reduces duplicate API calls.
_flight_search_cache: dict[tuple[str, str, str], tuple[FlightSearchResponse, float]] = {}
FLIGHT_SEARCH_CACHE_TTL_SEC = 120  # 2 minutes


def _flight_search_cache_key(origin: str, destination: str, date: str) -> tuple[str, str, str]:
    return ((origin or "").strip().lower(), (destination or "").strip().lower(), (date or "").strip()[:10])


def _city_to_iata(city: str) -> str | None:
    """Map city name to IATA airport code. Returns None if unknown."""
    if not city:
        return None
    key = (city or "").strip().lower().replace(",", " ")
    key = " ".join(key.split())
    if not key:
        return None
    if key in _CITY_TO_IATA:
        return _CITY_TO_IATA[key]
    # 3-letter IATA passed through
    if len(key) == 3 and key.isalpha():
        return key.upper()
    return None


def search_flights(origin: str, destination: str, date: str) -> FlightSearchResponse:
    """
    Search flights for a route on a given date using Aviation Stack.
    Maps city names to IATA. Returns reason/message when empty so frontend can show a hint and avoid retries.
    Results are cached for 2 minutes to prevent duplicate API calls.
    """
    key = _flight_search_cache_key(origin, destination, date)
    now = time.time()
    if key in _flight_search_cache:
        cached, expiry = _flight_search_cache[key]
        if now < expiry:
            return cached
        del _flight_search_cache[key]

    dep_iata = _city_to_iata(origin)
    arr_iata = _city_to_iata(destination)
    if not dep_iata or not arr_iata:
        out = FlightSearchResponse(
            flights=[],
            reason="city_not_found",
            message=f"Unknown city: origin '{origin}' or destination '{destination}'. Use a known city (e.g. Houston, Tokyo) or 3-letter IATA code.",
        )
    else:
        api_key = _get_api_key()
        if not api_key:
            out = FlightSearchResponse(
                flights=[],
                reason="no_api_key",
                message="Set AVIATION_STACK_API_KEY in .env for flight search. Get a key at https://aviationstack.com/",
            )
        else:
            # flight_date format YYYY-MM-DD
            flight_date = (date or "").strip()[:10]
            if len(flight_date) != 10 or flight_date[4] != "-" or flight_date[7] != "-":
                out = FlightSearchResponse(
                    flights=[],
                    reason="invalid_date",
                    message="Use date format YYYY-MM-DD (e.g. 2026-03-15).",
                )
            else:
                out = _search_flights_impl(api_key, dep_iata, arr_iata, flight_date, origin, destination)

    _flight_search_cache[key] = (out, now + FLIGHT_SEARCH_CACHE_TTL_SEC)
    return out


def _search_flights_impl(
    api_key: str, dep_iata: str, arr_iata: str, flight_date: str, origin: str, destination: str
) -> FlightSearchResponse:
    try:
        with httpx.Client(timeout=TIMEOUT_SEC) as client:
            r = client.get(
                f"{AVIATION_STACK_BASE}/flights",
                params={
                    "access_key": api_key,
                    "dep_iata": dep_iata,
                    "arr_iata": arr_iata,
                    "flight_date": flight_date,
                    "limit": 25,
                },
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, httpx.ConnectError, Exception):
        return FlightSearchResponse(
            flights=[],
            reason="api_error",
            message="Flight API request failed. Free tier may not support this route/date; try Aviation Stack dashboard.",
        )

    results = data.get("data")
    if not results or not isinstance(results, list):
        # Free tier often returns no data for future dates or certain routes
        return FlightSearchResponse(
            flights=[],
            reason="no_flights",
            message="No flights returned for this route/date. Aviation Stack free tier may only have real-time or limited data.",
        )

    flights: list[FlightSearchItem] = []
    for f in results[:25]:
        dep = f.get("departure") or {}
        arr = f.get("arrival") or {}
        flight_info = f.get("flight") or {}
        flight_iata = (flight_info.get("iata") or "").strip() or (flight_info.get("number") or "")
        if not flight_iata and flight_info.get("number"):
            airline = (f.get("airline") or {}).get("iata") or ""
            flight_iata = f"{airline}{flight_info.get('number')}"

        dep_airport = _format_airport(dep.get("airport"), dep.get("iata"))
        arr_airport = _format_airport(arr.get("airport"), arr.get("iata"))
        dep_scheduled = dep.get("scheduled") or dep.get("estimated") or ""
        arr_scheduled = arr.get("scheduled") or arr.get("estimated") or ""

        flights.append(
            FlightSearchItem(
                flight_iata=flight_iata or "—",
                departure=FlightSearchDeparture(
                    airport=dep_airport,
                    iata=(dep.get("iata") or "").strip() or dep_iata,
                    scheduled=dep_scheduled,
                ),
                arrival=FlightSearchArrival(
                    airport=arr_airport,
                    iata=(arr.get("iata") or "").strip() or arr_iata,
                    scheduled=arr_scheduled,
                ),
                status=_normalize_status(f.get("flight_status")),
            )
        )

    return FlightSearchResponse(flights=flights)


def track_flight(flight_number: str) -> FlightTrackResponse:
    """
    Track a flight by airline code + number (e.g. UA1234).
    Always returns a FlightTrackResponse; use status "invalid", "unavailable", or "not_found"
    when input is bad, key is missing, or no flight data.
    """
    empty_dep = FlightTrackDeparture(airport="", time="")
    empty_arr = FlightTrackArrival(airport="", time="")

    flight_iata = _parse_flight_number(flight_number)
    if not flight_iata:
        return FlightTrackResponse(status="invalid", departure=empty_dep, arrival=empty_arr)

    api_key = _get_api_key()
    if not api_key:
        return FlightTrackResponse(
            status="unavailable",
            departure=FlightTrackDeparture(airport="Configure AVIATION_STACK_API_KEY", time=""),
            arrival=empty_arr,
        )

    try:
        with httpx.Client(timeout=TIMEOUT_SEC) as client:
            r = client.get(
                f"{AVIATION_STACK_BASE}/flights",
                params={"access_key": api_key, "flight_iata": flight_iata, "limit": 1},
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, httpx.ConnectError, Exception):
        return FlightTrackResponse(status="unavailable", departure=empty_dep, arrival=empty_arr)

    results = data.get("data")
    if not results or not isinstance(results, list) or len(results) == 0:
        return FlightTrackResponse(status="not_found", departure=empty_dep, arrival=empty_arr)

    flight = results[0]
    dep = flight.get("departure") or {}
    arr = flight.get("arrival") or {}

    dep_airport = _format_airport(dep.get("airport"), dep.get("iata"))
    arr_airport = _format_airport(arr.get("airport"), arr.get("iata"))
    dep_time = dep.get("scheduled") or dep.get("estimated") or dep.get("actual") or ""
    arr_time = arr.get("scheduled") or arr.get("estimated") or arr.get("actual") or ""

    return FlightTrackResponse(
        status=_normalize_status(flight.get("flight_status")),
        departure=FlightTrackDeparture(airport=dep_airport, time=dep_time),
        arrival=FlightTrackArrival(airport=arr_airport, time=arr_time),
    )
