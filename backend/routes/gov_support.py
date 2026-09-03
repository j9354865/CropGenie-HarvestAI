"""
Government Agriculture Support route — POST /api/gov-support/ask
=================================================================

Answers farmer questions about Pakistani government agriculture schemes,
subsidies, loans, crop insurance, and support programs.

No datasets — the intelligence lives entirely in Gemini's general
knowledge, with a clear disclaimer that farmers should verify details
with their local Agriculture Extension Office or ZTBL.

Flow
----
1. Validate that the farmer's query is provided.
2. Build a structured prompt with the query and optional province.
3. Send the prompt to Gemini's dedicated gov-support model.
4. Return the answer alongside a disclaimer.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    GovSupportRequest,
    GovSupportResponse,
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


# ── Disclaimer (always included in every response) ───────────────────
DISCLAIMER = (
    "This info may not be the latest. "
    "Please check with your local Agriculture office "
    "or ZTBL bank to be sure."
)


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/gov-support/ask", response_model=GovSupportResponse)
async def ask_gov_support(request: GovSupportRequest):
    """
    Answer a farmer's question about government agriculture support.

    Covers topics like:
      - Subsidies (fertilizer, seed, equipment)
      - Low-interest agricultural loans (ZTBL, Kisan Card)
      - Crop insurance schemes
      - Provincial agriculture support programs
      - Eligibility and application guidance

    Accepts
    -------
    - query (required): the farmer's question
    - province (optional): Punjab, Sindh, KPK, or Balochistan
    """
    try:
        gemini_svc = _get_gemini_service()

        query = request.query.strip()
        province = request.province.strip() if request.province else None
        province_provided = province is not None and len(province) > 0

        # ── Step 1: Build the Gemini prompt ──────────────────────
        province_text = (
            province
            if province_provided
            else "not specified — give general Pakistan-wide info, "
                 "and mention that programs vary by province"
        )

        prompt = (
            f"Farmer's question: {query}\n"
            f"Province (if specified): {province_text}\n\n"
            f"Answer the farmer's question about government agriculture "
            f"support (this may include subsidies, low-interest loans, "
            f"crop insurance schemes, seed/fertilizer support programs, "
            f"or Kisan Card type initiatives).\n\n"
            f"Guidelines:\n"
            f"- Be clear and practical — explain in simple language a "
            f"farmer can understand\n"
            f"- If you are not fully certain about a specific current "
            f"scheme's exact details (amounts, deadlines, eligibility), "
            f"say so clearly and advise the farmer to confirm with their "
            f"local Agriculture Extension Office or Zarai Taraqiati Bank "
            f"(ZTBL) rather than stating uncertain details as fact\n"
            f"- Keep the response under 150 words\n"
            f"- Respond in simple, farmer-friendly language"
        )

        # ── Step 2: Ask Gemini for government support info ──────
        try:
            answer = gemini_svc.generate_gov_support_advice(prompt, language="en-US")
        except Exception as gemini_exc:
            # Gemini failed — return a friendly fallback
            logger.warning(
                "Gemini gov-support generation failed (%s), using fallback",
                gemini_exc,
            )
            answer = (
                "Unable to fetch information about government agriculture "
                "support right now. Please try again in a moment.\n\n"
                "In the meantime, you can contact your local Agriculture "
                "Extension Office or visit the nearest Zarai Taraqiati "
                "Bank (ZTBL) branch for information about current schemes "
                "and subsidies."
            )

        # ── Step 3: Build the response ──────────────────────────
        display_province = (
            province if province_provided else "General (all provinces)"
        )

        return GovSupportResponse(
            query=query,
            province=display_province,
            answer=answer,
            disclaimer=DISCLAIMER,
        )

    except ValueError as exc:
        logger.warning("Gov-support validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        logger.error("Gov-support service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected gov-support error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
