"""
Curated destinations for inspiration. Can be seeded from Hugging Face DeepNLP/travel-ai-agent
or a built-in list. Set DESTINATIONS_USE_HF=1 and have 'datasets' installed to try HF load.
"""

import os
from typing import Optional

from app.models import DestinationItem, DestinationsResponse

# Built-in curated list (no external API). Frontend can use for inspiration.
DEFAULT_DESTINATIONS: list[DestinationItem] = [
    DestinationItem(
        id="tokyo",
        name="Tokyo",
        country="Japan",
        query="Tokyo, Japan",
        image_url="https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=800",
        description="Blend of tradition and tech, temples and neon.",
    ),
    DestinationItem(
        id="paris",
        name="Paris",
        country="France",
        query="Paris, France",
        image_url="https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800",
        description="Art, culture, and iconic landmarks.",
    ),
    DestinationItem(
        id="london",
        name="London",
        country="United Kingdom",
        query="London, UK",
        image_url="https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800",
        description="History, museums, and diverse neighborhoods.",
    ),
    DestinationItem(
        id="barcelona",
        name="Barcelona",
        country="Spain",
        query="Barcelona, Spain",
        image_url="https://images.unsplash.com/photo-1583422409516-2895a77efded?w=800",
        description="Gaudí, beaches, and Mediterranean vibe.",
    ),
    DestinationItem(
        id="newyork",
        name="New York",
        country="USA",
        query="New York, USA",
        image_url="https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=800",
        description="City that never sleeps; arts, food, and skyline.",
    ),
    DestinationItem(
        id="dubai",
        name="Dubai",
        country="UAE",
        query="Dubai, UAE",
        image_url="https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=800",
        description="Modern luxury, desert, and culture.",
    ),
    DestinationItem(
        id="singapore",
        name="Singapore",
        country="Singapore",
        query="Singapore",
        image_url="https://images.unsplash.com/photo-1525625293386-3f8f99389edd?w=800",
        description="Gardens, food, and multicultural city.",
    ),
    DestinationItem(
        id="rome",
        name="Rome",
        country="Italy",
        query="Rome, Italy",
        image_url="https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=800",
        description="Ancient history, art, and cuisine.",
    ),
    DestinationItem(
        id="bali",
        name="Bali",
        country="Indonesia",
        query="Bali, Indonesia",
        image_url="https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=800",
        description="Beaches, temples, and wellness.",
    ),
    DestinationItem(
        id="istanbul",
        name="Istanbul",
        country="Turkey",
        query="Istanbul, Turkey",
        image_url="https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?w=800",
        description="Where East meets West; bazaars and mosques.",
    ),
]

# Trending = same list with a different ordering (e.g. by popularity); can be extended later.
TRENDING_IDS = ["tokyo", "bali", "paris", "barcelona", "london", "dubai", "singapore", "rome", "newyork", "istanbul"]


# Placeholder image for DeepNLP rows that don't have one
DESTINATION_PLACEHOLDER_IMAGE = "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400&h=300&fit=crop"


def _query_from_website(website: str | None) -> str:
    """Extract domain for query (e.g. mindtrip.ai)."""
    if not website:
        return ""
    s = str(website).strip().replace("https://", "").replace("http://", "")
    return s.split("/")[0] if s else ""


def _load_from_huggingface() -> list[DestinationItem] | None:
    """
    Load from DeepNLP/travel-ai-agent when datasets is installed.
    Set DESTINATIONS_USE_HF=1 to enable. Maps: content_name, publisher_id, subfield, website, description.
    """
    if os.environ.get("DESTINATIONS_USE_HF") != "1":
        return None
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("DeepNLP/travel-ai-agent", split="train")
        out: list[DestinationItem] = []
        for i, row in enumerate(list(ds)[:24]):  # limit 24 for UI
            name = str(row.get("content_name", "Unknown"))
            website = str(row.get("website", "")) if row.get("website") else ""
            query = _query_from_website(website) if website else name
            dest_id = str(row.get("publisher_id", f"dest-{i}"))
            country = str(row.get("subfield", "Travel"))
            desc_raw = (str(row.get("description", "")) or "").strip()
            description = desc_raw[:200] if desc_raw else None
            out.append(
                DestinationItem(
                    id=dest_id,
                    name=name,
                    country=country,
                    query=query,
                    image_url=DESTINATION_PLACEHOLDER_IMAGE,
                    description=description,
                )
            )
        return out if out else None
    except Exception:
        return None


def get_destinations(trending: bool = False) -> DestinationsResponse:
    """Return destinations list. If trending=True, return a trending-ordered subset."""
    hf = _load_from_huggingface()
    if hf:
        if trending:
            # Keep first 10 as "trending" order when from HF
            return DestinationsResponse(destinations=hf[:15])
        return DestinationsResponse(destinations=hf)

    if trending:
        by_id = {d.id: d for d in DEFAULT_DESTINATIONS}
        ordered = [by_id[i] for i in TRENDING_IDS if i in by_id]
        return DestinationsResponse(destinations=ordered)
    return DestinationsResponse(destinations=DEFAULT_DESTINATIONS)
