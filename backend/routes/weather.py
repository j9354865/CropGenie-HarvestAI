"""
Weather route — /api/weather

Fetches current weather + forecast from OpenWeatherMap, then asks the
Gemini AI service to generate practical farming advice based on the
conditions.  Returns everything in a single response.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import WeatherRequest, WeatherResponse
from backend.services.gemini_service import GeminiService
from backend.services.weather_service import WeatherService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised service instances
_weather_service: WeatherService | None = None
_gemini_service: GeminiService | None = None


def _get_weather_service() -> WeatherService:
    """Lazy-init the weather service so a missing API key surfaces clearly."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with the chat route)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


def _format_weather_summary(
    current,
    forecast,
) -> str:
    """
    Build a plain-text weather summary that the Gemini AI can interpret.

    This avoids sending raw JSON — a readable summary lets the model
    focus on actionable advice rather than parsing data.
    """
    lines = [
        f"Location: {current.city}",
        f"Current: {current.temperature}°C (feels like {current.feels_like}°C), "
        f"{current.description}",
        f"Humidity: {current.humidity}%, Wind: {current.wind_speed} m/s",
        "",
        "Forecast:",
    ]

    for day in forecast:
        lines.append(
            f"  {day.date}: {day.temp_min}–{day.temp_max}°C, "
            f"{day.description}, "
            f"rain chance {day.rain_chance:.0%}, "
            f"humidity {day.humidity}%"
        )

    return "\n".join(lines)


@router.post("/weather", response_model=WeatherResponse)
async def weather(request: WeatherRequest):
    """
    Fetch weather data and generate AI farming advice.

    Accepts either a city name or lat/lon coordinates.
    Returns current conditions, 3-day forecast, and AI-generated advice.
    """
    # Validate that at least one location type is provided
    if not request.city and (request.lat is None or request.lon is None):
        raise HTTPException(
            status_code=422,
            detail="Provide either a city name or latitude/longitude coordinates.",
        )

    try:
        weather_svc = _get_weather_service()
        gemini_svc = _get_gemini_service()

        # Step 1: Fetch current weather
        current = weather_svc.get_current_weather(
            city=request.city,
            lat=request.lat,
            lon=request.lon,
        )

        # Step 2: Fetch forecast
        forecast = weather_svc.get_forecast(
            city=request.city,
            lat=request.lat,
            lon=request.lon,
            days=3,
        )

        # Step 3: Generate AI farming advice from the weather data
        weather_summary = _format_weather_summary(current, forecast)
        advice = gemini_svc.generate_weather_advice(weather_summary, language="en-US")

        return WeatherResponse(
            current=current,
            forecast=forecast,
            advice=advice,
        )

    except ValueError as exc:
        # Bad location, invalid input — 400
        logger.warning("Weather request validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        # API failure (weather or Gemini) — 503
        logger.error("Weather service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected weather error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
