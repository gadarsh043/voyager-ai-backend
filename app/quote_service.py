"""
Build an in-depth quote from a selected itinerary option.
Itemized breakdown, summary (subtotal + platform_fee = total), and points optimization.
"""

from datetime import datetime

from app.models import (
    BreakdownItem,
    DailyPlan,
    ItineraryOption,
    PointsOptimization,
    QuoteBreakdown,
    QuoteResponse,
    QuoteSummary,
)

PLATFORM_FEE = 15.0


def _nights(check_in: str, check_out: str) -> int:
    try:
        a = datetime.strptime(check_in, "%Y-%m-%d")
        b = datetime.strptime(check_out, "%Y-%m-%d")
        return max(0, (b - a).days)
    except (ValueError, TypeError):
        return 1


def _build_breakdown(option: ItineraryOption) -> tuple[QuoteBreakdown, float]:
    """
    Parse daily_plan into itemized breakdown and return (breakdown, subtotal).
    Allocates (total_estimated_cost - platform_fee) across flights, hotels, activities.
    """
    plan: DailyPlan = option.daily_plan
    budget = max(0, float(option.total_estimated_cost) - PLATFORM_FEE)

    flights: list[BreakdownItem] = []
    if plan.flight_from_source:
        leg = plan.flight_from_source
        desc = f"Outbound: {leg.from_location} → {leg.to_location or 'destination'}"
        flights.append(BreakdownItem(description=desc, amount=0))  # amount set below
    if plan.flight_to_origin:
        leg = plan.flight_to_origin
        desc = f"Return: {leg.from_location} → {leg.to_location or leg.from_location}"
        flights.append(BreakdownItem(description=desc, amount=0))

    hotels: list[BreakdownItem] = []
    if plan.hotel_stay:
        for stay in plan.hotel_stay:
            n = _nights(stay.check_in, stay.check_out) or 1
            hotels.append(
                BreakdownItem(
                    description=f"{stay.name}, {n} night{'s' if n != 1 else ''}",
                    amount=0,
                )
            )

    activities: list[BreakdownItem] = []
    for day in plan.days:
        for act in day.activities:
            activities.append(
                BreakdownItem(
                    description=f"Day {day.day}: {act.start_from} ({act.time_to_spend})",
                    amount=0,
                )
            )

    # Allocate budget: ~45% flights, ~35% hotels, ~20% activities
    n_flights = len(flights) or 1
    n_hotels = len(hotels) or 1
    n_activities = len(activities) or 1
    flight_share = 0.45
    hotel_share = 0.35
    activity_share = 0.20

    flight_total = budget * flight_share
    hotel_total = budget * hotel_share
    activity_total = budget * activity_share

    for i, f in enumerate(flights):
        f.amount = round(flight_total / n_flights, 2)
    for i, h in enumerate(hotels):
        h.amount = round(hotel_total / n_hotels, 2)
    for i, a in enumerate(activities):
        a.amount = round(activity_total / n_activities, 2)

    subtotal = sum(f.amount for f in flights) + sum(h.amount for h in hotels) + sum(a.amount for a in activities)
    # Normalize to match budget if rounding drifted
    if subtotal > 0 and abs(subtotal - budget) > 0.01:
        scale = budget / subtotal
        for f in flights:
            f.amount = round(f.amount * scale, 2)
        for h in hotels:
            h.amount = round(h.amount * scale, 2)
        for a in activities:
            a.amount = round(a.amount * scale, 2)
        subtotal = sum(f.amount for f in flights) + sum(h.amount for h in hotels) + sum(a.amount for a in activities)

    return QuoteBreakdown(flights=flights, hotels=hotels, activities=activities), round(subtotal, 2)


def _build_points_optimization(option: ItineraryOption, total: float) -> PointsOptimization:
    """Trip-agnostic points suggestions; can be replaced with LLM later."""
    return PointsOptimization(
        best_card_to_use="Amex Gold (4x on Dining/Flights)",
        potential_points_earned=round(total * 4, 0),  # e.g. 4x on travel
        suggestions=[
            "Use a travel rewards card for flight and hotel bookings to earn 3–5x points.",
            "Book flights directly with the airline when possible for better award availability.",
            "Consider splitting payment: one card for flights, another for dining to maximize category bonuses.",
        ],
        transfer_partners=[
            "Chase → United, Hyatt, Southwest",
            "Amex → Delta, Marriott, Hilton",
            "Citi → American Airlines, JetBlue",
        ],
        redemption_tips=[
            "Check award space 2–4 weeks before your dates for better availability.",
            "For this itinerary, compare cash vs. points; sometimes paid fares are better value.",
            "Hotel redemptions often offer better point value than flights on this route.",
        ],
    )


def build_quote(option: ItineraryOption) -> QuoteResponse:
    """
    Analyze the selected plan and return an in-depth quote:
    itemized breakdown, summary (subtotal + platform_fee = total), and points optimization.
    """
    breakdown, subtotal = _build_breakdown(option)
    total = round(subtotal + PLATFORM_FEE, 2)
    per_person = round(total / 2, 2)  # default 2 travelers; frontend can override with num_persons if added later

    summary = QuoteSummary(
        subtotal=subtotal,
        platform_fee=PLATFORM_FEE,
        total=total,
        per_person=per_person,
    )

    points_optimization = _build_points_optimization(option, total)

    return QuoteResponse(
        breakdown=breakdown,
        summary=summary,
        points_optimization=points_optimization,
    )
