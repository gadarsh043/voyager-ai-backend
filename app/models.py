"""
Pydantic models for itinerary API request and response.
Matches the exact JSON spec for the frontend.
"""

from typing import Optional, Union

from pydantic import BaseModel


# ----- Request -----


class TripParams(BaseModel):
    """Trip parameters from the frontend form. All optional; defaults used when empty."""

    origin: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date_range: Optional[str] = None
    budget: Optional[Union[int, float, str]] = None
    preferences: Optional[list[str]] = None

    class Config:
        extra = "ignore"


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


class Activity(BaseModel):
    start_from: str
    start_time: str
    reach_time: str
    time_to_spend: str
    image_url: Optional[str] = None
    google_maps_url: Optional[str] = None


class DayPlan(BaseModel):
    day: int
    activities: list[Activity]


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


# ----- Quote (in-depth quote from selected plan) -----


class QuoteRequest(BaseModel):
    """Request body for POST /itinerary/quote: the selected option from /itinerary/generate."""

    option: ItineraryOption


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
