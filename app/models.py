"""
Pydantic models for itinerary API request and response.
Matches the exact JSON spec for the frontend.
"""

from typing import Optional, Union

from pydantic import AliasChoices, BaseModel, Field


# ----- Request -----


class TripParams(BaseModel):
    """Trip parameters from the frontend form. All optional; defaults used when empty. Use per_person_budget and num_persons for cost (no separate budget field)."""

    origin: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date_range: Optional[str] = None
    per_person_budget: Optional[Union[int, float]] = None
    num_persons: Optional[int] = None
    accommodation_type: Optional[str] = None  # e.g. hotel, hostel, apartment
    pace: Optional[str] = None  # e.g. relaxed, moderate, fast
    preferences: Optional[list[str]] = None
    disability: Optional[bool] = None
    dietary: Optional[bool] = None

    class Config:
        extra = "ignore"


# ----- Flight tracking -----


class FlightTrackRequest(BaseModel):
    """Request body for POST /flights/track."""

    flight_number: str  # e.g. UA1234, AA456


class FlightTrackDeparture(BaseModel):
    airport: str
    time: str  # ISO8601


class FlightTrackArrival(BaseModel):
    airport: str
    time: str  # ISO8601


class FlightTrackResponse(BaseModel):
    """Response for POST /flights/track. status: scheduled, active, landed, cancelled, etc."""

    status: str
    departure: FlightTrackDeparture
    arrival: FlightTrackArrival


# ----- Flight search (Plan page) -----


class FlightSearchRequest(BaseModel):
    """Request body for POST /flights/search."""

    origin: str  # city name, e.g. Houston
    destination: str  # city name, e.g. Tokyo
    date: str  # YYYY-MM-DD, e.g. 2026-03-15


class FlightSearchDeparture(BaseModel):
    airport: str
    iata: str
    scheduled: str  # ISO8601


class FlightSearchArrival(BaseModel):
    airport: str
    iata: str
    scheduled: str  # ISO8601


class FlightSearchItem(BaseModel):
    flight_iata: str
    departure: FlightSearchDeparture
    arrival: FlightSearchArrival
    status: str


class FlightSearchResponse(BaseModel):
    flights: list[FlightSearchItem]
    # When flights are empty, reason helps the frontend show a message and avoid retrying
    reason: Optional[str] = None  # e.g. no_api_key, city_not_found, no_flights, invalid_date, api_error
    message: Optional[str] = None  # Human-readable hint


# ----- Response: nested structures -----


class FlightLeg(BaseModel):
    from_location: str
    to_location: Optional[str] = None
    start_time: str
    reach_by: str


class HotelStay(BaseModel):
    name: str
    check_in: str
    check_out: str
    image_url: Optional[str] = None
    google_maps_url: Optional[str] = None
    price_level: Optional[int] = None       # Google Places price_level (0-4)
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    description: Optional[str] = None


class Activity(BaseModel):
    start_from: str
    start_time: str
    reach_time: str
    time_to_spend: str
    name: Optional[str] = None  # e.g. pick label in plan-with-picks
    image_url: Optional[str] = None
    google_maps_url: Optional[str] = None
    rating: Optional[float] = None          # Google Places rating (1–5)
    user_ratings_total: Optional[int] = None  # number of Google reviews
    description: Optional[str] = None       # editorial_summary or vicinity
    price_level: Optional[int] = None       # Google Places price_level (0-4)


class DayPlan(BaseModel):
    day: int
    activities: list[Activity]          # Where to Go
    places_to_eat: list[Activity] = []  # Where to Eat


class DailyPlan(BaseModel):
    flight_from_source: Optional[FlightLeg] = None
    flight_to_origin: Optional[FlightLeg] = None
    hotel_stay: Optional[list[HotelStay]] = None
    days: list[DayPlan]


class ItineraryOption(BaseModel):
    id: str
    label: str
    daily_plan: DailyPlan
    total_estimated_cost: float


class ItineraryGenerateResponse(BaseModel):
    options: list[ItineraryOption]
    suggested_days_for_trip: Optional[int] = None  # LLM or backend suggestion; e.g. 5 for a short city break


# ----- Quote (in-depth quote from selected plan) -----


class QuoteRequest(BaseModel):
    """Request body for POST /itinerary/quote: the selected option from /itinerary/generate."""

    option: ItineraryOption
    num_persons: Optional[int] = None  # optional; used to compute per_person


class BreakdownItem(BaseModel):
    description: str
    amount: float


class QuoteBreakdown(BaseModel):
    flights: list[BreakdownItem] = []
    hotels: list[BreakdownItem] = []
    activities: list[BreakdownItem] = []


class QuoteSummary(BaseModel):
    subtotal: float
    platform_fee: float
    total: float
    per_person: Optional[float] = None


class PointsOptimization(BaseModel):
    best_card_to_use: str
    potential_points_earned: float
    suggestions: list[str] = []
    transfer_partners: list[str] = []
    redemption_tips: list[str] = []


class QuoteResponse(BaseModel):
    breakdown: QuoteBreakdown
    summary: QuoteSummary
    points_optimization: PointsOptimization


# ----- Trip document (single AI-generated full document) -----


class TripDocumentQuote(BaseModel):
    """Quote data as sent by frontend when requesting a trip document."""

    subtotal: float
    total: float
    breakdown: QuoteBreakdown
    points_optimization: PointsOptimization


class TripDocumentRequest(BaseModel):
    """Request body for POST /itinerary/trip-document."""

    option: ItineraryOption
    quote: TripDocumentQuote
    origin: str
    destination: str


class TripDocumentResponse(BaseModel):
    """Response: full trip document as a single string (plain text or markdown)."""

    content: str


# ----- Plan from picks (single itinerary built from user picks) -----


class PickItem(BaseModel):
    """One pick: label and optional Google Maps link."""

    label: str
    google_maps_url: Optional[str] = None


class PlanWithPicksRequest(BaseModel):
    """Request body for POST /itinerary/plan-with-picks."""

    picks: list[PickItem]
    origin: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PlanWithPicksResponse(BaseModel):
    """Response: one option built from picks, same shape as one element of /itinerary/generate options."""

    option_id: str
    option: ItineraryOption


# ----- Destinations (inspiration) -----


class DestinationItem(BaseModel):
    """Frontend expects "image" in JSON (see BACKEND_PROMPT / destinations spec)."""

    id: str
    name: str
    country: str
    query: str  # e.g. "Tokyo, Japan" or domain like "mindtrip.ai"
    image_url: Optional[str] = Field(
        None,
        serialization_alias="image",
        validation_alias=AliasChoices("image", "image_url"),
    )
    description: Optional[str] = None


class DestinationsResponse(BaseModel):
    destinations: list[DestinationItem]
