"""
Crop Grafting Advisor route — POST /api/grafting/advise
=======================================================

Advises farmers on crop/plant grafting (paiwand) — which crops or
varieties can be successfully grafted together, why, and how to do it.

No datasets or ML models — the intelligence lives entirely in Gemini's
general horticultural knowledge, guided by a carefully structured prompt
that produces extremely simple, farmer-friendly advice.

Flow
----
1. Validate that the base_crop is provided.
2. Build a structured prompt with the crop, goal, and language.
3. Send the prompt to Gemini's dedicated grafting model.
4. Return the advice alongside the input parameters.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    GraftingRequest,
    GraftingResponse,
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


# ── Supported language codes ─────────────────────────────────────────
# Maps short codes to human-readable language names for the prompt.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ur": "Urdu (اردو)",
    "ps": "Pashto (پښتو)",
}

# Default language if an unsupported code is passed
DEFAULT_LANGUAGE = "en"


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/grafting/advise", response_model=GraftingResponse)
async def advise_grafting(request: GraftingRequest):
    """
    Generate practical grafting (paiwand) advice for a farmer's crop.

    Covers:
      - Compatible crops/varieties for grafting
      - Best grafting method and simple steps
      - Best timing (season/month)
      - Practical benefits
      - Common mistakes to avoid

    Accepts
    -------
    - base_crop (required): the plant/tree the farmer currently has
    - goal (optional): what the farmer wants to achieve
    - language (optional): 'en', 'ur', or 'ps' — defaults to English
    """
    try:
        gemini_svc = _get_gemini_service()

        # ── Step 1: Normalise inputs ─────────────────────────────
        base_crop = request.base_crop.strip()

        goal = request.goal.strip() if request.goal else None
        goal_text = (
            goal
            if goal
            else "general improvement — suggest the most useful grafting options"
        )

        # Resolve language — fall back to English for unsupported codes
        lang_code = request.language.strip().lower()
        if lang_code not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Unsupported language '%s', defaulting to English",
                lang_code,
            )
            lang_code = DEFAULT_LANGUAGE

        lang_name = SUPPORTED_LANGUAGES[lang_code]

        # ── Step 2: Build the Gemini prompt ──────────────────────
        prompt = (
            f"Farmer's current crop/tree: {base_crop}\n"
            f"Farmer's goal: {goal_text}\n\n"
            f"Provide clear, practical grafting advice. Your response "
            f"MUST be written in extremely simple, plain language that "
            f"ANY farmer can understand regardless of education level — "
            f"avoid technical botanical jargon; if you must use a "
            f"technical term, immediately explain it in one simple phrase.\n\n"
            f"Cover these points, each as a short, clearly labeled section:\n"
            f"1. Compatible crops/varieties — which specific varieties or "
            f"related plants can be grafted onto or with this crop, and why "
            f"they are compatible (same plant family, similar growth pattern, "
            f"etc. — explained simply)\n"
            f"2. Best method — which grafting technique fits this combination "
            f"(e.g. cleft grafting, bud grafting, whip grafting) explained "
            f"in simple steps\n"
            f"3. Best timing — what season or time of year is ideal for "
            f"this grafting\n"
            f"4. Benefits — what practical benefit the farmer gets (better "
            f"taste, disease resistance, faster fruit-bearing, drought "
            f"tolerance, etc.)\n"
            f"5. Things to avoid — common mistakes or incompatible "
            f"combinations to watch out for\n\n"
            f"Keep the entire response under 200 words total.\n"
            f"Write in {lang_name}.\n"
            f"Use short sentences and simple everyday words throughout — "
            f"imagine explaining this to a farmer with no formal education, "
            f"not a student or scientist."
        )

        # ── Step 3: Ask Gemini for grafting advice ───────────────
        try:
            advice = gemini_svc.generate_grafting_advice(prompt, language=request.language)
        except Exception as gemini_exc:
            # Gemini failed — return a friendly fallback
            logger.warning(
                "Gemini grafting advice failed (%s), using fallback",
                gemini_exc,
            )
            advice = (
                "We could not give grafting advice right now. "
                "Please try again in a moment.\n\n"
                "For now, ask a local farming expert or nursery owner "
                "for help with grafting "
                f"**{base_crop}**."
            )

        # ── Step 4: Build the response ───────────────────────────
        display_goal = goal if goal else "General improvement"

        return GraftingResponse(
            base_crop=base_crop,
            goal=display_goal,
            language=lang_code,
            advice=advice,
        )

    except ValueError as exc:
        logger.warning("Grafting validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        logger.error("Grafting service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected grafting error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
