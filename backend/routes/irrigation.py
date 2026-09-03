"""
Smart Irrigation route — POST /api/irrigation/recommend
=======================================================

Tells farmers WHEN and HOW MUCH to irrigate their crops by combining:
  • Live weather data (temperature, humidity, 3-day rain forecast)
  • Gemini AI reasoning (crop-specific irrigation logic)

No local datasets or ML models needed — the intelligence lives entirely
in the weather API + Gemini's agricultural knowledge.

Flow
----
1. Validate crop_name and location are provided.
2. Fetch current weather + 3-day forecast from OpenWeatherMap.
3. Build a structured prompt with real values (crop, weather, rain,
   soil type, days since last irrigation).
4. Send the prompt to Gemini's dedicated irrigation model.
5. Parse the response into structured fields (decision, reason,
   water_amount) and return the full advice alongside.
"""

import logging
import re

from fastapi import APIRouter, HTTPException

from backend.models.schemas import IrrigationRequest, IrrigationResponse
from backend.services.gemini_service import GeminiService
from backend.services.weather_service import WeatherService

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy-initialised services ────────────────────────────────────────
_weather_service: WeatherService | None = None
_gemini_service: GeminiService | None = None


def _get_weather_service() -> WeatherService:
    """Lazy-init the weather service."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


# ── Helpers ───────────────────────────────────────────────────────────

def _format_rain_forecast(forecast_days) -> str:
    """
    Turn a list of ForecastDay objects into a readable 3-day rain summary.

    Example output:
        "Day 1 (2026-09-02): 10% rain chance | Day 2 (2026-09-03): 45% rain chance | Day 3 (2026-09-04): 20% rain chance"
    """
    parts = []
    for i, day in enumerate(forecast_days, 1):
        pct = day.rain_chance * 100  # convert 0-1 to percentage
        parts.append(f"Day {i} ({day.date}): {pct:.0f}% rain chance")
    return " | ".join(parts) if parts else "No forecast data available"


def _parse_irrigation_response(raw_text: str) -> dict:
    """
    Extract structured fields from Gemini's irrigation response text.

    Tries to find:
      - decision: "Irrigate now", "Wait", or "Irrigate in X days"
      - reason: a one-sentence explanation
      - water_amount: "light", "moderate", or "heavy"

    If parsing fails, returns sensible defaults so the UI never breaks.

    Parameters
    ----------
    raw_text : str
        The full text response from Gemini.

    Returns
    -------
    dict
        Keys: decision, reason, water_amount
    """
    text_lower = raw_text.lower()

    # ── Decision ────────────────────────────────────────────────
    decision = "Irrigate now"  # safe default

    if "irrigate now" in text_lower or "water now" in text_lower:
        decision = "Irrigate now"
    elif re.search(r"irrigate in \d+ day", text_lower):
        match = re.search(r"irrigate in (\d+) day", text_lower)
        days = match.group(1) if match else "2"
        decision = f"Irrigate in {days} days"
    elif "wait" in text_lower or "no need" in text_lower or "skip" in text_lower:
        decision = "Wait"

    # ── Water amount ────────────────────────────────────────────
    water_amount = "moderate"  # safe default
    if "light" in text_lower and "water" in text_lower:
        water_amount = "light"
    elif "heavy" in text_lower and "water" in text_lower:
        water_amount = "heavy"
    elif "moderate" in text_lower and "water" in text_lower:
        water_amount = "moderate"

    # If the decision is "Wait", water amount isn't relevant
    if decision == "Wait":
        water_amount = "N/A"

    # ── Reason (first sentence that isn't the decision) ─────────
    # Split into sentences and find a reason-like line
    sentences = re.split(r'[.!\n]', raw_text)
    reason = ""
    for s in sentences:
        s = s.strip()
        if len(s) > 10 and s.lower() != decision.lower():
            reason = s
            break
    if not reason:
        reason = "Based on current weather conditions and crop needs."

    return {
        "decision": decision,
        "reason": reason,
        "water_amount": water_amount,
    }


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/irrigation/recommend", response_model=IrrigationResponse)
async def irrigation_recommend(request: IrrigationRequest):
    """
    Get a smart irrigation recommendation for a crop at a given location.

    Combines live OpenWeatherMap data (temperature, humidity, 3-day rain
    forecast) with Gemini AI reasoning to tell the farmer:
      - Whether to irrigate now, wait, or irrigate in X days
      - Why (one simple sentence)
      - How much water to apply (light / moderate / heavy)

    Accepts
    -------
    - crop_name (required): e.g. "Wheat", "Cotton", "Rice"
    - location (required): city name, e.g. "Multan"
    - last_irrigation_days_ago (optional): days since last watering
    - soil_type (optional): e.g. "Loamy", "Clayey", "Sandy"
    """
    try:
        weather_svc = _get_weather_service()
        gemini_svc = _get_gemini_service()

        crop_name = request.crop_name.strip()
        location = request.location.strip()

        # ── Step 1: Fetch current weather + 3-day forecast ─────────
        current = weather_svc.get_current_weather(city=location)
        forecast = weather_svc.get_forecast(city=location, days=3)

        temperature = current.temperature
        humidity = current.humidity
        rain_forecast_text = _format_rain_forecast(forecast)

        logger.info(
            "Irrigation weather for %s: %.1f°C, %d%% humidity, forecast: %s",
            location, temperature, humidity, rain_forecast_text,
        )

        # ── Step 2: Build the Gemini prompt with real values ───────
        soil_text = request.soil_type.strip() if request.soil_type else "not specified"
        last_irr_text = (
            f"{request.last_irrigation_days_ago} days ago"
            if request.last_irrigation_days_ago is not None
            else "unknown"
        )

        prompt = (
            f"Crop: {crop_name}\n"
            f"Location: {location}\n"
            f"Soil type: {soil_text}\n"
            f"Current temperature: {temperature:.1f}°C\n"
            f"Humidity: {humidity}%\n"
            f"Rain forecast (next 3 days): {rain_forecast_text}\n"
            f"Days since last irrigation: {last_irr_text}\n\n"
            f"Based on this data, provide a short, practical irrigation "
            f"recommendation. Your response must include:\n"
            f"1. A clear decision: 'Irrigate now' or 'Wait' or "
            f"'Irrigate in X days'\n"
            f"2. A one-sentence reason in simple language\n"
            f"3. If irrigating: approximate water amount guidance "
            f"(light/moderate/heavy watering)\n\n"
            f"Keep the entire response under 60 words. "
            f"Respond in simple, farmer-friendly language."
        )

        # ── Step 3: Ask Gemini for irrigation advice ──────────────
        try:
            raw_advice = gemini_svc.generate_irrigation_advice(prompt, language="en-US")
        except Exception as gemini_exc:
            # Gemini failed — return a friendly fallback so we never crash
            logger.warning(
                "Gemini irrigation advice failed (%s), using fallback",
                gemini_exc,
            )
            raw_advice = (
                f"Unable to generate detailed advice right now. "
                f"Current conditions: {temperature:.1f}°C, {humidity}% humidity. "
                f"Rain forecast: {rain_forecast_text}. "
                f"Please try again in a moment."
            )

        # ── Step 4: Parse the AI response into structured fields ──
        parsed = _parse_irrigation_response(raw_advice)

        return IrrigationResponse(
            crop=crop_name,
            location=current.city,  # resolved city name from weather API
            decision=parsed["decision"],
            reason=parsed["reason"],
            water_amount=parsed["water_amount"],
            raw_advice=raw_advice,
            temperature=temperature,
            humidity=humidity,
            rain_forecast_text=rain_forecast_text,
        )

    except ValueError as exc:
        # Invalid location / missing input
        logger.warning("Irrigation validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        # Weather API or Gemini service failure
        logger.error("Irrigation service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        # Catch-all — never crash
        logger.error("Unexpected irrigation error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
