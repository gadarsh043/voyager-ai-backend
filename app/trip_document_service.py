"""
Generate a single full trip document for PDF: exact section headings, plain text only.
Sections: TRIP ITINERARY, OUTBOUND FLIGHT, HOTELS, DAILY ACTIVITIES, RETURN FLIGHT,
SUGGESTIONS, CURRENCY USAGE, MOBILE PLAN, CARD BENEFITS, LOCAL LANGUAGE CHEAT SHEET, EMERGENCY CONTACTS.
Uses request origin/destination for language and emergency contacts.
"""

import json

import httpx

from app.models import TripDocumentRequest, TripDocumentResponse

from app.ollama_service import get_ollama_model, OLLAMA_CHAT

# Longer timeout for full-document generation.
DOCUMENT_TIMEOUT_SEC = 60.0


def _resolve_origin_destination(req: TripDocumentRequest) -> tuple[str, str]:
    """
    Use origin and destination from the request as source of truth.
    Fall back to option.daily_plan only when the frontend sends empty values.
    """
    origin = (req.origin or "").strip()
    destination = (req.destination or "").strip()
    plan = req.option.daily_plan
    if not origin and plan.flight_from_source:
        origin = (plan.flight_from_source.from_location or "").strip()
    if not destination:
        if plan.flight_from_source and plan.flight_from_source.to_location:
            destination = (plan.flight_from_source.to_location or "").strip()
        elif plan.flight_to_origin:
            destination = (plan.flight_to_origin.from_location or "").strip()
    return origin or "Origin", destination or "Destination"


def _build_document_prompt(req: TripDocumentRequest) -> str:
    """Build a single user prompt so the LLM outputs the exact section headings for PDF styling."""
    option = req.option
    plan = option.daily_plan
    origin, destination = _resolve_origin_destination(req)
    quote = req.quote
    po = quote.points_optimization

    outbound = ""
    if plan.flight_from_source:
        leg = plan.flight_from_source
        outbound = f"from_location={leg.from_location}, to_location={leg.to_location or 'destination'}, start_time={leg.start_time}, reach_by={leg.reach_by}"

    return_flight = ""
    if plan.flight_to_origin:
        leg = plan.flight_to_origin
        return_flight = f"from_location={leg.from_location}, to_location={leg.to_location or 'origin'}, start_time={leg.start_time}, reach_by={leg.reach_by}"

    hotels_lines = []
    if plan.hotel_stay:
        for h in plan.hotel_stay:
            hotels_lines.append(f"name={h.name}, check_in={h.check_in}, check_out={h.check_out}")

    daily_activities_lines = []
    for day in plan.days:
        daily_activities_lines.append(f"Day {day.day}:")
        for a in day.activities:
            name_part = f", name={a.name}" if getattr(a, "name", None) else ""
            daily_activities_lines.append(
                f"  - start_from={a.start_from}, start_time={a.start_time}, time_to_spend={a.time_to_spend}{name_part}"
            )

    hotel_name = plan.hotel_stay[0].name if plan.hotel_stay else "N/A"

    return f"""Generate a trip document as PLAIN TEXT only (no markdown, no HTML). The document will be turned into a PDF. Use the EXACT section headings below so the PDF renderer can style them. Use "---" as a separator between major sections. Keep each section under its heading.

ORIGIN (use for language and emergency contacts): {origin}
DESTINATION (use for language, currency, mobile, and emergency contacts): {destination}
PLAN LABEL: {option.label}

Rule: Use ONLY the origin and destination above for CURRENCY USAGE, MOBILE PLAN, LOCAL LANGUAGE CHEAT SHEET, and EMERGENCY CONTACTS. Do NOT infer from the itinerary (e.g. Dallas–Houston → English, USD; Japan → Japanese, 110/119).

Output exactly these 11 sections IN THIS ORDER:

1. TRIP ITINERARY
Summary and high-level itinerary: origin → destination, and the plan label. One short paragraph.

2. OUTBOUND FLIGHT
From the option's daily_plan. Use this data: {outbound or 'None'}
Format: from_location, to_location, start_time, reach_by on clear lines.

3. HOTELS
From daily_plan.hotel_stay. Use this data:
{chr(10).join(hotels_lines) if hotels_lines else 'None'}
Format: name, check_in, check_out for each hotel.

4. DAILY ACTIVITIES
From daily_plan.days. For each day: a line "Day N" then bullet lines for each activity (start_from, start_time, time_to_spend, name if present).
Data:
{chr(10).join(daily_activities_lines)}

5. RETURN FLIGHT
From daily_plan.flight_to_origin. Use this data: {return_flight or 'None'}
Format: from_location, to_location, start_time, reach_by.

6. SUGGESTIONS
From the quote's points_optimization: suggestions, redemption_tips, best_card_to_use, potential_points_earned, transfer_partners.
Suggestions: {json.dumps(po.suggestions)}
Redemption tips: {json.dumps(po.redemption_tips)}
Best card: {po.best_card_to_use}
Potential points: {po.potential_points_earned}
Transfer partners: {json.dumps(po.transfer_partners)}

7. CURRENCY USAGE
Short, practical tips for DESTINATION = {destination}: local currency name and code, cards, ATMs. Use the destination to choose (e.g. US → USD; Japan → JPY).

8. MOBILE PLAN
Short tips for {destination}: roaming, eSIM, offline maps.

9. CARD BENEFITS
Best card and points info from the quote: {po.best_card_to_use}, potential_points_earned={po.potential_points_earned}.

10. LOCAL LANGUAGE CHEAT SHEET
Phrases in the DESTINATION's main language. Use DESTINATION = {destination} to choose (e.g. US city → English; Japan → Japanese; France → French). Include: Hello, Thank you, Please, Yes, No, Where is…?, How much?. Use romanization and local script if not English. Do NOT assume Japanese unless destination implies Japan.

11. EMERGENCY CONTACTS
MUST be generated from the DESTINATION (and optionally traveller origin). Use REAL local emergency numbers for the destination country (e.g. 911 in US, 112 in EU, 110 police and 119 fire/ambulance in Japan, 999 in UK). Include:
• Local emergency number(s) for the destination country — use the real number(s) for that country.
• Embassy/Consulate — tell the user to add their country's embassy or consulate in the destination; give a short instruction e.g. "Search: [destination] embassy [traveller's country]".
• Hotel front desk — instruct to save the hotel's 24h number when they check in. Optionally mention the hotel name from the itinerary if available: {hotel_name}.
• Travel insurance — instruct to add policy number and 24h hotline.
• Emergency contact at home — instruct to add a name and phone number.

Output only the trip document text. No preamble, no "Here is your document". Use the exact headings: TRIP ITINERARY, OUTBOUND FLIGHT, HOTELS, DAILY ACTIVITIES, RETURN FLIGHT, SUGGESTIONS, CURRENCY USAGE, MOBILE PLAN, CARD BENEFITS, LOCAL LANGUAGE CHEAT SHEET, EMERGENCY CONTACTS."""


async def generate_trip_document(req: TripDocumentRequest) -> TripDocumentResponse:
    """
    Call Ollama once to generate the full trip document. Returns content or raises for 4xx/5xx.
    """
    user_content = _build_document_prompt(req)

    try:
        async with httpx.AsyncClient(timeout=DOCUMENT_TIMEOUT_SEC) as client:
            r = await client.post(
                OLLAMA_CHAT,
                json={
                    "model": get_ollama_model(),
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a travel assistant. Output only the requested trip document in plain text. No markdown, no HTML, no code blocks. Use the exact section headings given (e.g. TRIP ITINERARY, EMERGENCY CONTACTS) so the PDF renderer can style them. Emergency contacts must use real local emergency numbers for the destination country.",
                        },
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                },
            )
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError as e:
        raise TripDocumentError("AI service unavailable. Please try again later.") from e
    except httpx.HTTPStatusError as e:
        raise TripDocumentError(f"AI service error: {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise TripDocumentError("AI service request failed.") from e

    content = (data.get("message") or {}).get("content") or ""
    if isinstance(content, dict):
        content = json.dumps(content)
    content = content.strip()
    if not content:
        raise TripDocumentError("AI returned an empty document.")

    return TripDocumentResponse(content=content)


class TripDocumentError(Exception):
    """Raised when trip document generation fails (AI unavailable or empty)."""

    pass