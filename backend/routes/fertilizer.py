"""
Fertilizer recommendation route — /api/recommend-fertilizer

Combines an ML-based fertilizer classification prediction with a
Gemini-powered farmer-friendly explanation.  Optionally fetches current
temperature and humidity from the Weather service when a city is provided.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import FertilizerRequest, FertilizerResponse
from backend.services.fertilizer_service import FertilizerService
from backend.services.gemini_service import GeminiService
from backend.services.weather_service import WeatherService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised services
_fertilizer_service: FertilizerService | None = None
_gemini_service: GeminiService | None = None
_weather_service: WeatherService | None = None


def _get_fertilizer_service() -> FertilizerService:
    """Lazy-init the fertilizer ML service."""
    global _fertilizer_service
    if _fertilizer_service is None:
        _fertilizer_service = FertilizerService()
    return _fertilizer_service


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


def _get_weather_service() -> WeatherService:
    """Lazy-init the weather service."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


@router.post("/recommend-fertilizer", response_model=FertilizerResponse)
async def recommend_fertilizer(request: FertilizerRequest):
    """
    Predict the best fertilizer and generate a farmer-friendly explanation.

    1. Optionally fetch current temperature/humidity from the weather API
    2. Run ML classification to predict the recommended fertilizer
    3. Pass prediction + context to Gemini for an explanation
    4. Return the combined result
    """
    try:
        fert_svc = _get_fertilizer_service()
        gemini_svc = _get_gemini_service()

        # ── Step 1: Resolve temperature and humidity ─────────────────
        temperature = request.temperature
        humidity = request.humidity

        if request.city and (temperature is None or humidity is None):
            try:
                weather_svc = _get_weather_service()
                current_weather = weather_svc.get_current_weather(
                    city=request.city
                )
                if temperature is None:
                    temperature = current_weather.temperature
                if humidity is None:
                    humidity = current_weather.humidity
                logger.info(
                    "Weather lookup for %s: T=%.1f°C, H=%d%%",
                    request.city,
                    temperature,
                    humidity,
                )
            except Exception as weather_exc:
                logger.warning(
                    "Weather lookup failed (%s), using manual defaults",
                    weather_exc,
                )

        # Fallback defaults if still None
        if temperature is None:
            temperature = 30.0    # reasonable default for South Asia
        if humidity is None:
            humidity = 55.0       # reasonable default

        # ── Step 2: ML prediction ────────────────────────────────────
        fertilizer_name = fert_svc.predict(
            temperature=temperature,
            humidity=humidity,
            moisture=request.moisture,
            soil_type=request.soil_type,
            crop_type=request.crop,
            nitrogen=request.nitrogen,
            phosphorus=request.phosphorus,
            potassium=request.potassium,
        )
        logger.info("ML predicted fertilizer: %s", fertilizer_name)

        # ── Step 3: Gemini explanation ───────────────────────────────
        try:
            explanation = gemini_svc.generate_fertilizer_explanation(
                fertilizer_name=fertilizer_name,
                crop=request.crop,
                soil_type=request.soil_type,
                nitrogen=request.nitrogen,
                phosphorus=request.phosphorus,
                potassium=request.potassium,
                acres=request.acres,
                temperature=temperature,
                humidity=humidity,
                moisture=request.moisture,
                language=request.language,
            )
        except Exception as gemini_exc:
            # Gemini failure is non-critical — fall back to a basic message
            logger.warning(
                "Gemini explanation failed (%s), using fallback", gemini_exc
            )
            explanation = (
                f"We suggest using **{fertilizer_name}** for your "
                f"{request.crop} crop on {request.soil_type} soil.\n\n"
                f"Use about 50–60 kg per acre at sowing time. "
                f"Total for {request.acres} acres: "
                f"{50 * request.acres:.0f}–{60 * request.acres:.0f} kg.\n\n"
                "⚠️ AI advice could not load right now — "
                "ask a local farming expert for help."
            )

        return FertilizerResponse(
            fertilizer_name=fertilizer_name,
            crop=request.crop,
            soil_type=request.soil_type,
            nitrogen=request.nitrogen,
            phosphorus=request.phosphorus,
            potassium=request.potassium,
            acres=request.acres,
            temperature=temperature,
            humidity=humidity,
            moisture=request.moisture,
            explanation=explanation,
        )

    except ValueError as exc:
        # Bad crop/soil name, invalid input — 400
        logger.warning("Fertilizer request validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        # Model not loaded, API failure — 503
        logger.error("Fertilizer service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected fertilizer error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )


@router.get("/fertilizer-options")
async def fertilizer_options():
    """Return the supported crops, soil types, and fertilizer names."""
    fert_svc = _get_fertilizer_service()
    return {
        "crops": fert_svc.get_supported_crops(),
        "soil_types": fert_svc.get_supported_soils(),
        "fertilizers": fert_svc.get_supported_fertilizers(),
    }
