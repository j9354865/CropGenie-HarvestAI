"""
Profit prediction route — /api/predict-profit

Combines ML-based crop yield prediction with weather-based risk analysis
and profit estimation.  Orchestrates the YieldService and WeatherService.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import PredictProfitRequest, ProfitResponse
from backend.services.weather_service import WeatherService
from backend.services.yield_service import YieldService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised services
_yield_service: YieldService | None = None
_weather_service: WeatherService | None = None


def _get_yield_service() -> YieldService:
    """Lazy-init the yield prediction service."""
    global _yield_service
    if _yield_service is None:
        _yield_service = YieldService()
    return _yield_service


def _get_weather_service() -> WeatherService:
    """Lazy-init the weather service."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


@router.post("/predict-profit", response_model=ProfitResponse)
async def predict_profit(request: PredictProfitRequest):
    """
    Predict crop yield, calculate profit, and assess weather risk.

    1. Runs ML prediction using soil/fertilizer features + current temp
    2. Calculates revenue/cost/profit using crop economics reference data
    3. Fetches weather forecast to detect risk conditions
    4. Returns everything in a single response
    """
    try:
        yield_svc = _get_yield_service()
        weather_svc = _get_weather_service()

        # ── Step 1: Fetch weather for temperature + risk analysis ───
        try:
            current_weather = weather_svc.get_current_weather(
                city=request.city
            )
            forecast = weather_svc.get_forecast(
                city=request.city, days=3
            )
            current_temp = current_weather.temperature
            current_humidity = current_weather.humidity
            weather_city = current_weather.city

            # Max rain chance across the forecast window
            max_rain = max(
                (f.rain_chance for f in forecast), default=0.0
            )
        except Exception as weather_exc:
            # Weather failure is non-critical — fall back to request inputs
            logger.warning(
                "Weather lookup failed (%s), using defaults", weather_exc
            )
            current_temp = 32.0      # reasonable default for South Asia
            current_humidity = 50
            weather_city = request.city
            max_rain = 0.0
            forecast = []

        # ── Step 2: ML yield prediction ─────────────────────────────
        predicted_yield = yield_svc.predict_yield(
            fertilizer=request.fertilizer,
            temp=current_temp,
            n=request.nitrogen,
            p=request.phosphorus,
            k=request.potassium,
        )

        total_yield = round(predicted_yield * request.acres, 2)

        # ── Step 3: Profit calculation ──────────────────────────────
        profit_data = yield_svc.calculate_profit(
            predicted_yield=predicted_yield,
            crop=request.crop,
            acres=request.acres,
        )

        # ── Step 4: Weather risk assessment ─────────────────────────
        risk_warning = yield_svc.assess_weather_risk(
            crop=request.crop,
            rain_chance_max=max_rain,
            humidity=current_humidity,
            temperature=current_temp,
        )

        return ProfitResponse(
            crop=request.crop,
            acres=request.acres,
            predicted_yield=predicted_yield,
            total_yield=total_yield,
            revenue=profit_data["revenue"],
            cost=profit_data["cost"],
            profit=profit_data["profit"],
            profit_margin=profit_data["profit_margin"],
            risk_warning=risk_warning,
            has_risk=risk_warning is not None,
            weather_city=weather_city,
            temperature=current_temp,
            humidity=current_humidity,
        )

    except ValueError as exc:
        # Bad crop name, invalid input — 400
        logger.warning("Profit prediction validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        # Model not loaded, API failure — 503
        logger.error("Profit prediction service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected profit prediction error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )


@router.get("/crops")
async def list_supported_crops():
    """Return the list of crops supported for profit prediction."""
    from backend.services.crop_economics import SUPPORTED_CROPS
    return {"crops": SUPPORTED_CROPS}
