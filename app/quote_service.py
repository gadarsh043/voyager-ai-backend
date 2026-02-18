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


def _trip_context(option: ItineraryOption) -> dict:
    """Extract trip context from the selected option for personalized points_optimization."""
    plan = option.daily_plan
    nights = 0
    hotel_names: list[str] = []
    if plan.hotel_stay:
        for stay in plan.hotel_stay:
            n = _nights(stay.check_in, stay.check_out) or 1
            nights += n
            hotel_names.append(stay.name)
    num_days = len(plan.days)
    num_activities = sum(len(d.activities) for d in plan.days)
    cost = float(option.total_estimated_cost)
    label = (option.label or "").lower()
    return {
        "nights": nights or 1,
        "hotel_names": hotel_names,
        "num_days": num_days or 1,
        "num_activities": num_activities or 1,
        "cost": cost,
        "is_budget": "budget" in label,
        "is_luxury": "luxury" in label,
    }


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
    """Points suggestions tied to the selected plan: cost, nights, hotels, and plan type."""
    ctx = _trip_context(option)
    potential = round(total * 4, 0)

    # Card and suggestions vary by plan type and trip size
    if ctx["is_luxury"] or ctx["cost"] >= 4000:
        best_card = "Amex Platinum (5x on flights/hotels, lounge access)"
        potential = round(total * 5, 0)
        suggestions = [
            f"For your ${ctx['cost']:,.0f} trip, use the same card for flights and hotels to hit sign-up or category bonuses.",
            "Book your stay directly with the hotel to earn 5x and elite benefits.",
            "Consider using points for the return flight to save cash for experiences.",
        ]
        redemption_tips = [
            "With multiple nights at " + (ctx["hotel_names"][0] if ctx["hotel_names"] else "your hotel") + ", check Marriott/Hilton award charts for this stay.",
            "Luxury redemptions often have better value when booked 2–4 weeks out.",
        ]
    elif ctx["is_budget"] or ctx["cost"] < 2000:
        best_card = "Chase Sapphire Preferred (2x travel, flexible redemptions)"
        potential = round(total * 2, 0)
        suggestions = [
            f"For this {ctx['nights']}-night trip (${ctx['cost']:,.0f}), put flights and hotel on one travel card to maximize points.",
            "If you have a card with no foreign transaction fee, use it for all overseas spend.",
            "Save 1.25–1.5¢/point by redeeming through Chase Travel for this itinerary.",
        ]
        redemption_tips = [
            "For shorter trips, cash fares can beat points; compare before booking.",
            f"Your {ctx['num_activities']} activities and dining will add points if you use a dining card.",
        ]
    else:
        best_card = "Amex Gold (4x on Dining/Flights)"
        suggestions = [
            f"For your {ctx['nights']}-night trip (${ctx['cost']:,.0f}), use one card for flights and one for dining to maximize category bonuses.",
            "Book flights directly with the airline when possible for better award availability.",
            "Put hotel and activities on the same card to track travel spend in one place.",
        ]
        redemption_tips = [
            "Check award space 2–4 weeks before your dates for this itinerary.",
            "Compare cash vs. points for your outbound and return legs; mix if one is better value.",
        ]
        if ctx["hotel_names"]:
            redemption_tips.append(f"Hotel redemptions for {ctx['hotel_names'][0]} may offer good value on this route.")

    return PointsOptimization(
        best_card_to_use=best_card,
        potential_points_earned=potential,
        suggestions=suggestions,
        transfer_partners=[
            "Chase → United, Hyatt, Southwest",
            "Amex → Delta, Marriott, Hilton",
            "Citi → American Airlines, JetBlue",
        ],
        redemption_tips=redemption_tips,
    )


def build_quote(option: ItineraryOption, num_persons: int | None = None) -> QuoteResponse:
    """
    Analyze the selected plan and return an in-depth quote:
    itemized breakdown, summary (subtotal + platform_fee = total), and points optimization.
    num_persons: optional; used for per_person (default 2).
    """
    breakdown, subtotal = _build_breakdown(option)
    total = round(subtotal + PLATFORM_FEE, 2)
    n = max(1, num_persons) if num_persons is not None else 2
    per_person = round(total / n, 2)

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
