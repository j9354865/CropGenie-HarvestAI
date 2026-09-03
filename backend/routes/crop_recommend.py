"""
Crop recommendation route — /api/crop/recommend

Combines an ML-based crop classification prediction with a
Gemini-powered farmer-friendly explanation of WHY the recommended
crop suits the given soil and climate conditions.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    CropAlternative,
    CropRecommendationRequest,
    CropRecommendationResponse,
)
from backend.services.crop_recommendation_service import CropRecommendationService
from backend.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised services (same pattern as fertilizer.py)
_crop_service: CropRecommendationService | None = None
_gemini_service: GeminiService | None = None


def _get_crop_service() -> CropRecommendationService:
    """Lazy-init the crop recommendation ML service."""
    global _crop_service
    if _crop_service is None:
        _crop_service = CropRecommendationService()
    return _crop_service


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


@router.post("/crop/recommend", response_model=CropRecommendationResponse)
async def recommend_crop(request: CropRecommendationRequest):
    """
    Predict the best crop to grow and generate a farmer-friendly explanation.

    1. Run ML classification to predict the recommended crop + confidence
    2. Pass prediction + soil/climate context to Gemini for an explanation
    3. Return the combined result
    """
    try:
        crop_svc = _get_crop_service()
        gemini_svc = _get_gemini_service()

        # ── Step 1: ML prediction ────────────────────────────────────
        prediction = crop_svc.predict(
            nitrogen=request.nitrogen,
            phosphorus=request.phosphorus,
            potassium=request.potassium,
            temperature=request.temperature,
            humidity=request.humidity,
            ph=request.ph,
            rainfall=request.rainfall,
        )

        recommended_crop = prediction["recommended_crop"]
        confidence = prediction["confidence"]
        top_3 = prediction["top_3"]

        logger.info(
            "ML predicted crop: %s (confidence=%.4f)", recommended_crop, confidence
        )

        # ── Step 2: Gemini explanation ───────────────────────────────
        try:
            explanation = gemini_svc.generate_crop_recommendation_explanation(
                recommended_crop=recommended_crop,
                confidence=confidence,
                top_3_alternatives=top_3,
                nitrogen=request.nitrogen,
                phosphorus=request.phosphorus,
                potassium=request.potassium,
                temperature=request.temperature,
                humidity=request.humidity,
                ph=request.ph,
                rainfall=request.rainfall,
                language="en-US",
            )
        except Exception as gemini_exc:
            # Gemini failure is non-critical — fall back to a basic message
            logger.warning(
                "Gemini explanation failed (%s), using fallback", gemini_exc
            )
            explanation = (
                f"We suggest growing **{recommended_crop}** — "
                f"it looks best for your soil and weather.\n\n"
                f"Your soil: N={request.nitrogen}, P={request.phosphorus}, "
                f"K={request.potassium}, pH={request.ph}\n"
                f"Your weather: {request.temperature}°C, "
                f"{request.humidity}% humidity, "
                f"{request.rainfall}mm rain\n\n"
                "⚠️ AI advice could not load right now — "
                "ask a local farming expert for help."
            )

        # ── Step 3: Build response ───────────────────────────────────
        return CropRecommendationResponse(
            recommended_crop=recommended_crop,
            confidence=confidence,
            top_3_alternatives=[
                CropAlternative(crop=alt["crop"], confidence=alt["confidence"])
                for alt in top_3
            ],
            explanation=explanation,
        )

    except ValueError as exc:
        # Bad input values — 400
        logger.warning("Crop recommendation validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        # Model not loaded, service failure — 503
        logger.error("Crop recommendation service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected crop recommendation error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )


@router.get("/crop/supported-crops")
async def supported_crops():
    """Return the list of crops the model can recommend."""
    crop_svc = _get_crop_service()
    return {"crops": crop_svc.get_supported_crops()}
