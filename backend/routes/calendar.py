"""
Farming Calendar route — POST /api/calendar/generate
=====================================================

Generates a complete, personalised farming schedule for a given crop
and location using Gemini AI reasoning only.

No datasets, no external weather/market APIs — the intelligence lives
entirely in Gemini's agricultural knowledge base.

Flow
----
1. Validate crop_name and location are provided.
2. Build a structured prompt with the crop, location, and optional
   sowing date.
3. Send the prompt to Gemini's dedicated calendar model.
4. Parse the response into structured sections (sowing, watering,
   fertilizer, spray reminders, harvesting) and return the full
   advice alongside.
"""

import logging
import re

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    CalendarRequest,
    CalendarResponse,
    CalendarSchedule,
)
from backend.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy-initialised service ─────────────────────────────────────────
_gemini_service: GeminiService | None = None


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


# ── Response parser ───────────────────────────────────────────────────

def _parse_calendar_response(raw_text: str) -> CalendarSchedule:
    """
    Extract structured calendar sections from Gemini's calendar text.

    Tries to find section headers like "Sowing", "Watering", "Fertilizer",
    "Spray", "Harvest" and collects the lines beneath each.

    If parsing fails for any section, the entire raw text is placed in
    ``sowing`` so the UI can still display ``raw_advice``.

    Parameters
    ----------
    raw_text : str
        The full text response from Gemini.

    Returns
    -------
    CalendarSchedule
        Structured calendar object.
    """
    lines = raw_text.strip().split("\n")

    # Section detection helpers
    def _is_header(line: str, keywords: list[str]) -> bool:
        """
        Check if a line is a section header matching any keyword.

        To avoid false positives on content bullets that happen to
        contain words like "water" or "seed", we require headers to:
          - Start with a number marker (1., 2., etc.) or # heading, OR
          - Be short and colon-free (e.g. "**Sowing Window**")
        """
        low = line.lower().strip("*#•- ")
        has_keyword = any(kw in low for kw in keywords)
        if not has_keyword:
            return False

        # Path 1: numbered section like "1. Sowing Window" or "## 3. Fertilizer"
        if re.match(r"^\d+[\.\)]", low) or line.lstrip().startswith("#"):
            return True

        # Path 2: short keyword line with no colon (not a content bullet)
        if len(low) < 50 and ":" not in low:
            return True

        return False

    def _collect_bullets(section_lines: list[str]) -> list[str]:
        """Clean up bullet lines, stripping leading markers and bold."""
        result = []
        for ln in section_lines:
            # Remove bold markdown first, then leading bullet markers
            clean = ln.replace("**", "")
            # Strip bullet prefix but preserve ordinal digits (1st, 2nd, etc.)
            clean = re.sub(r"^[•\-*\s]+", "", clean).strip()
            # Remove a leading number-dot only if followed by a space (list marker)
            clean = re.sub(r"^\d+\.\s+", "", clean).strip()
            if clean and len(clean) > 3:
                result.append(clean)
        return result

    # Split into sections
    sowing_lines: list[str] = []
    watering_lines: list[str] = []
    fertilizer_lines: list[str] = []
    spray_lines: list[str] = []
    harvest_lines: list[str] = []

    current = None  # which section we're collecting into

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if _is_header(stripped, ["sowing", "sow", "planting", "seed"]):
            current = "sowing"
            continue
        elif _is_header(stripped, ["water", "irrigat"]):
            current = "watering"
            continue
        elif _is_header(stripped, ["fertil", "nutrient", "npk"]):
            current = "fertilizer"
            continue
        elif _is_header(stripped, ["spray", "pest", "disease", "fungic", "insect"]):
            current = "spray"
            continue
        elif _is_header(stripped, ["harvest", "reap", "maturity", "cutting"]):
            current = "harvest"
            continue

        # Collect the line into the current section
        if current == "sowing":
            sowing_lines.append(stripped)
        elif current == "watering":
            watering_lines.append(stripped)
        elif current == "fertilizer":
            fertilizer_lines.append(stripped)
        elif current == "spray":
            spray_lines.append(stripped)
        elif current == "harvest":
            harvest_lines.append(stripped)

    # Build the structured result
    sowing_text = " ".join(sowing_lines) if sowing_lines else "See full advice below"
    harvest_text = " ".join(harvest_lines) if harvest_lines else "See full advice below"

    watering_bullets = _collect_bullets(watering_lines)
    fertilizer_bullets = _collect_bullets(fertilizer_lines)
    spray_bullets = _collect_bullets(spray_lines)

    # Fallback: if nothing was parsed at all, use raw text as sowing
    if (
        not watering_bullets
        and not fertilizer_bullets
        and not spray_bullets
        and sowing_text == "See full advice below"
    ):
        sowing_text = raw_text.strip()

    return CalendarSchedule(
        sowing=sowing_text,
        watering_schedule=watering_bullets,
        fertilizer_schedule=fertilizer_bullets,
        spray_reminders=spray_bullets,
        harvesting=harvest_text,
    )


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/calendar/generate", response_model=CalendarResponse)
async def generate_calendar(request: CalendarRequest):
    """
    Generate a personalised farming calendar for a crop and location.

    Uses Gemini AI to produce a complete schedule covering:
      - Sowing window/date
      - Watering schedule (key irrigation stages)
      - Fertilizer schedule (type and timing)
      - Pest/disease spray reminders
      - Expected harvesting date/window

    Accepts
    -------
    - crop_name (required): e.g. "Wheat", "Cotton", "Rice"
    - location (required): region, district, or province
    - sowing_date (optional): planned sowing date string
    """
    try:
        gemini_svc = _get_gemini_service()

        crop_name = request.crop_name.strip()
        location = request.location.strip()
        sowing_date = request.sowing_date.strip() if request.sowing_date else None
        sowing_date_provided = sowing_date is not None and len(sowing_date) > 0

        # ── Step 1: Build the Gemini prompt ──────────────────────
        sowing_text = (
            sowing_date
            if sowing_date_provided
            else "not specified — suggest the ideal sowing window for this crop and region"
        )

        prompt = (
            f"Crop: {crop_name}\n"
            f"Region/Location: {location}\n"
            f"Planned sowing date: {sowing_text}\n\n"
            f"Generate a complete farming calendar for this crop in "
            f"this region. Structure your response as a clear schedule "
            f"with these sections, each with specific timing (use "
            f"relative timing like 'Day 0', 'Week 2', or actual dates "
            f"if a sowing date was given):\n\n"
            f"1. Sowing window/date\n"
            f"2. Watering schedule (key irrigation stages, not daily — "
            f"major stages like germination, flowering, maturity)\n"
            f"3. Fertilizer schedule (what type and roughly when, in stages)\n"
            f"4. Pest/disease spray reminders (critical vulnerable "
            f"growth stages)\n"
            f"5. Expected harvesting date/window\n\n"
            f"Keep it practical and realistic for the given region's "
            f"typical climate. Respond in simple, farmer-friendly "
            f"language. Keep the total response under 200 words, "
            f"formatted as a numbered or bulleted timeline."
        )

        # ── Step 2: Ask Gemini for calendar advice ──────────────
        try:
            raw_advice = gemini_svc.generate_calendar_advice(prompt, language="en-US")
        except Exception as gemini_exc:
            # Gemini failed — return a friendly fallback
            logger.warning(
                "Gemini calendar generation failed (%s), using fallback",
                gemini_exc,
            )
            raw_advice = (
                f"Unable to generate a farming calendar for {crop_name} "
                f"in {location} right now. Please try again in a moment.\n\n"
                f"For reference, typical {crop_name} farming in {location} "
                f"involves sowing, regular irrigation at key growth stages, "
                f"balanced fertiliser application, pest monitoring, and "
                f"harvesting when the crop reaches maturity."
            )

        # ── Step 3: Parse into structured sections ──────────────
        calendar = _parse_calendar_response(raw_advice)

        return CalendarResponse(
            crop=crop_name,
            location=location,
            sowing_date_provided=sowing_date_provided,
            calendar=calendar,
            raw_advice=raw_advice,
        )

    except ValueError as exc:
        logger.warning("Calendar validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        logger.error("Calendar service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected calendar error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
