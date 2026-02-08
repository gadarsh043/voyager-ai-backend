"""
Generate a single full trip document (itinerary, suggestions, currency, mobile, card benefits, language).
One Ollama call; returns plain-text content for PDF/print.
"""

import json

import httpx

from app.models import TripDocumentRequest, TripDocumentResponse

from app.ollama_service import DEFAULT_MODEL, OLLAMA_CHAT

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
    """Build a single user prompt with all inputs for the LLM."""
    option = req.option
    plan = option.daily_plan
    origin, destination = _resolve_origin_destination(req)
    quote = req.quote
    po = quote.points_optimization

    itinerary_lines = []
    if plan.flight_from_source:
        leg = plan.flight_from_source
        itinerary_lines.append(
            f"Outbound flight: {leg.from_location} → {leg.to_location or 'destination'} "
            f"depart {leg.start_time} arrive {leg.reach_by}"
        )
    if plan.hotel_stay:
        for h in plan.hotel_stay:
            itinerary_lines.append(f"Hotel: {h.name}, check-in {h.check_in}, check-out {h.check_out}")
    for day in plan.days:
        itinerary_lines.append(f"Day {day.day}:")
        for a in day.activities:
            itinerary_lines.append(
                f"  - {a.start_from}: {a.start_time}–{a.reach_time}, {a.time_to_spend}"
            )
    if plan.flight_to_origin:
        leg = plan.flight_to_origin
        itinerary_lines.append(
            f"Return flight: {leg.from_location} → {leg.to_location or 'origin'} "
            f"depart {leg.start_time} arrive {leg.reach_by}"
        )

    suggestions_text = json.dumps(po.suggestions + po.redemption_tips)
    breakdown_text = json.dumps(
        {
            "flights": [{"description": x.description, "amount": x.amount} for x in quote.breakdown.flights],
            "hotels": [{"description": x.description, "amount": x.amount} for x in quote.breakdown.hotels],
            "activities": [{"description": x.description, "amount": x.amount} for x in quote.breakdown.activities],
        },
        indent=2,
    )

    return f"""Create a single trip document in plain text for a traveler. Use clear section headings in ALL CAPS and separate sections with a blank line or "---". Do not use HTML.

---
ORIGIN (use this for language/currency/mobile): {origin}
DESTINATION (use this for language/currency/mobile): {destination}
---

**Rule:** For sections 3 (CURRENCY USAGE), 4 (MOBILE PLAN), and 6 (LOCAL LANGUAGE CHEAT SHEET), use ONLY the origin and destination above. Do NOT infer from the itinerary (flights/hotels/cities in the itinerary may be from another flow). E.g. Dallas–Houston → English, USD, US roaming; Japan → Japanese, JPY, etc.

Include exactly these 6 sections:

1. TRIP ITINERARY (in detail)
Use the itinerary data below only for structure, times, and places. Format it clearly.
{chr(10).join(itinerary_lines)}

2. SUGGESTIONS
Use these points/rewards suggestions and add 1–2 short practical tips for {destination}:
{suggestions_text}

3. CURRENCY USAGE
Use DESTINATION = {destination} (and origin {origin} if relevant). Write 3–5 short paragraphs: local currency name and code, card acceptance, ATM tips, notifying the bank, and whether to carry cash. Be specific to this destination only (e.g. US cities → USD; Japan → JPY).

4. MOBILE PLAN
Use DESTINATION = {destination}. Short practical tips: roaming vs local SIM/eSIM, data plans, offline maps, region-specific advice (e.g. US domestic vs Japan).

5. CARD BENEFITS
Use this data and add 1–2 sentences on maximizing value for this trip (flights/hotels/dining):
- Best card to use: {po.best_card_to_use}
- Potential points earned: {po.potential_points_earned}
- Transfer partners: {json.dumps(po.transfer_partners)}

6. LOCAL LANGUAGE CHEAT SHEET
Use DESTINATION = {destination} only to choose the language (e.g. Dallas/Houston/US city → English; Japan → Japanese; France → French). List 8–12 useful phrases. Format: phrase (romanization if not English) — local script if helpful. Include: Hello / Goodbye, Thank you / Please, Yes / No, How much? / Where is…? / I'd like…, Emergency / Help.

Breakdown for context (subtotal {quote.subtotal}, total {quote.total}):
{breakdown_text}

Output only the trip document text, no preamble or "Here is your document"."""


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
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a travel assistant. Output only the requested trip document in plain text with clear section headings. No HTML, no code blocks, no meta commentary.",
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