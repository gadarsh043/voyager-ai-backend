"""
Build one full itinerary (daily_plan) from user picks.
Orders picks into a day-by-day schedule and produces the same option shape as /itinerary/generate.
"""

from datetime import datetime

from app.models import (
    Activity,
    DailyPlan,
    DayPlan,
    FlightLeg,
    HotelStay,
    ItineraryOption,
    PickItem,
    PlanWithPicksRequest,
    PlanWithPicksResponse,
)

OPTION_ID_PREFIX = "picks_"
DEFAULT_OPTION_LABEL = "Your custom plan"


def _num_days(start: str | None, end: str | None) -> int:
    if not start or not end:
        return 1
    try:
        a = datetime.strptime(start, "%Y-%m-%d")
        b = datetime.strptime(end, "%Y-%m-%d")
        return max(1, (b - a).days + 1)
    except (ValueError, TypeError):
        return 1


def _nights(check_in: str, check_out: str) -> int:
    try:
        a = datetime.strptime(check_in, "%Y-%m-%d")
        b = datetime.strptime(check_out, "%Y-%m-%d")
        return max(0, (b - a).days)
    except (ValueError, TypeError):
        return 1


def _distribute_picks(picks: list[PickItem], num_days: int) -> list[list[PickItem]]:
    """Split picks across days in user order. Earlier days get more if uneven."""
    if num_days <= 0:
        num_days = 1
    if not picks:
        return [[] for _ in range(num_days)]
    per_day: list[list[PickItem]] = [[] for _ in range(num_days)]
    for i, p in enumerate(picks):
        per_day[i % num_days].append(p)
    return per_day


def _activity_from_pick(
    pick: PickItem,
    prev_label: str,
    start_h: int,
    start_m: int,
    duration_mins: int,
) -> Activity:
    """Build one Activity from a pick with placeholder times."""
    reach_m = start_m + duration_mins
    reach_h = start_h + reach_m // 60
    reach_m = reach_m % 60
    return Activity(
        start_from=prev_label,
        start_time=f"{start_h:02d}:{start_m:02d}",
        reach_time=f"{reach_h:02d}:{reach_m:02d}",
        time_to_spend=f"{duration_mins // 60}h {duration_mins % 60}m" if duration_mins >= 60 else f"{duration_mins}m",
        name=pick.label,
        google_maps_url=pick.google_maps_url,
    )


def _estimate_cost(
    has_flights: bool,
    nights: int,
    num_activities: int,
) -> float:
    """Rough total: flights, hotel, activities."""
    total = 0.0
    if has_flights:
        total += 400.0 * 2  # outbound + return
    if nights > 0:
        total += 120.0 * nights
    total += 80.0 * num_activities
    return round(total, 2)


def build_plan_from_picks(req: PlanWithPicksRequest) -> PlanWithPicksResponse:
    """
    Build one itinerary option from picks. Orders picks in user order across days.
    Uses origin/destination/start_date/end_date when provided for flights and hotel.
    """
    picks = req.picks or []
    origin = (req.origin or "").strip()
    destination = (req.destination or "").strip()
    start_date = (req.start_date or "").strip()
    end_date = (req.end_date or "").strip()

    num_days = _num_days(start_date or None, end_date or None)
    if not start_date:
        start_date = "2026-06-01"
    if not end_date:
        end_date = start_date

    # Flights and hotel only when we have origin/destination and dates
    flight_from_source: FlightLeg | None = None
    flight_to_origin: FlightLeg | None = None
    hotel_stay: list[HotelStay] = []
    if origin and destination and start_date and end_date:
        o_code = origin.upper()[:3] if len(origin) >= 2 else "ORIG"
        d_code = destination.upper()[:3] if len(destination) >= 2 else "DEST"
        flight_from_source = FlightLeg(
            from_location=o_code,
            start_time=f"{start_date}T08:00:00Z",
            reach_by=f"{start_date}T14:00:00Z",
        )
        flight_to_origin = FlightLeg(
            from_location=d_code,
            to_location=o_code,
            start_time=f"{end_date}T18:00:00Z",
            reach_by=f"{end_date}T23:00:00Z",
        )
        hotel_stay = [
            HotelStay(
                name=f"Hotel in {destination}",
                check_in=start_date,
                check_out=end_date,
            )
        ]

    # Distribute picks across days (user order)
    picks_per_day = _distribute_picks(picks, num_days)

    days: list[DayPlan] = []
    for d in range(1, num_days + 1):
        day_picks = picks_per_day[d - 1] if d <= len(picks_per_day) else []
        activities: list[Activity] = []
        prev_label = "Hotel" if hotel_stay else "Start"
        start_h, start_m = 9, 0
        for pick in day_picks:
            duration = 90  # 1h 30m default per activity
            act = _activity_from_pick(pick, prev_label, start_h, start_m, duration)
            activities.append(act)
            prev_label = pick.label
            start_m += duration
            start_h += start_m // 60
            start_m = start_m % 60
        if not activities:
            # Empty day: one placeholder activity
            activities = [
                Activity(
                    start_from=prev_label,
                    start_time="09:00",
                    reach_time="12:00",
                    time_to_spend="Free time",
                    name="Explore",
                )
            ]
        days.append(DayPlan(day=d, activities=activities))

    daily_plan = DailyPlan(
        flight_from_source=flight_from_source,
        flight_to_origin=flight_to_origin,
        hotel_stay=hotel_stay if hotel_stay else None,
        days=days,
    )

    num_activities = sum(len(d.activities) for d in days)
    nights = _nights(start_date, end_date) if (start_date and end_date) else (len(hotel_stay) * num_days if hotel_stay else 0)
    if hotel_stay and nights <= 0:
        nights = max(1, num_days - 1)
    total_estimated_cost = _estimate_cost(
        has_flights=flight_from_source is not None,
        nights=nights,
        num_activities=num_activities,
    )

    option_id = f"{OPTION_ID_PREFIX}1"
    option = ItineraryOption(
        id=option_id,
        label=DEFAULT_OPTION_LABEL,
        daily_plan=daily_plan,
        total_estimated_cost=total_estimated_cost,
    )

    return PlanWithPicksResponse(option_id=option_id, option=option)
