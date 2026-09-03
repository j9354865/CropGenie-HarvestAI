"""
Location insights route — /api/location-insights

Resolves a city/district in Pakistan to its agro-ecological zone,
fetches current weather, and asks Gemini for region-specific farming
guidance.  Returns zone characteristics, weather context, and
AI-generated insights in a single response.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import LocationInsightsRequest, LocationInsightsResponse
from backend.services.gemini_service import GeminiService
from backend.services.location_service import LocationService
from backend.services.weather_service import WeatherService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised services
_location_service: LocationService | None = None
_gemini_service: GeminiService | None = None
_weather_service: WeatherService | None = None


def _get_location_service() -> LocationService:
    """Lazy-init the location service."""
    global _location_service
    if _location_service is None:
        _location_service = LocationService()
    return _location_service


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


@router.post("/location-insights", response_model=LocationInsightsResponse)
async def location_insights(request: LocationInsightsRequest):
    """
    Resolve a city to its agro-ecological zone and generate farming insights.

    1. Look up the city in the Pakistan zone mapping
    2. Optionally fetch current weather for the city
    3. Pass zone characteristics + weather to Gemini for insights
    4. Return structured response with zone data, weather, and AI advice

    Graceful fallbacks at every step — never crashes.
    """
    try:
        loc_svc = _get_location_service()
        gemini_svc = _get_gemini_service()

        city = request.city.strip()

        # ── Step 1: Resolve zone ───────────────────────────────────
        zone_result = loc_svc.resolve_zone(city)

        if zone_result is not None:
            matched_district, zone_key, zone_data = zone_result
        else:
            # No zone match — use a generic "Unknown Zone" and let
            # Gemini reason about the area from weather data alone.
            logger.info(
                "No zone match for '%s', using weather-based fallback", city
            )
            matched_district = city.title()
            zone_key = "Unknown"
            zone_data = {
                "name": "Unknown (weather-based analysis)",
                "icon": "📍",
                "provinces": city.title(),
                "climate": "Data from live weather lookup",
                "soil": "To be determined based on weather patterns",
                "crops": ["Various — AI will suggest based on climate"],
                "irrigation": "To be determined based on local conditions",
                "challenges": [
                    "Location not in zone database",
                    "AI will provide general guidance",
                ],
            }

        # ── Step 2: Fetch current weather (optional, non-critical) ─
        temperature = None
        humidity = None
        weather_desc = None

        try:
            weather_svc = _get_weather_service()
            current = weather_svc.get_current_weather(city=city)
            temperature = current.temperature
            humidity = current.humidity
            weather_desc = current.description
            logger.info(
                "Weather for %s: %.1f°C, %d%%, %s",
                city, temperature, humidity, weather_desc,
            )
        except Exception as weather_exc:
            logger.warning(
                "Weather lookup failed for '%s' (%s), proceeding without weather",
                city,
                weather_exc,
            )

        # ── Step 3: Generate AI insights ───────────────────────────
        zone_fmt = loc_svc.format_zone_for_gemini(zone_data)

        try:
            insights = gemini_svc.generate_location_insights(
                zone_name=zone_fmt["zone_name"],
                climate=zone_fmt["climate"],
                soil=zone_fmt["soil"],
                crops=zone_fmt["crops"],
                irrigation=zone_fmt["irrigation"],
                challenges=zone_fmt["challenges"],
                city=city,
                temperature=temperature,
                humidity=humidity,
                weather_desc=weather_desc,
                language=request.language,
            )
        except Exception as gemini_exc:
            logger.warning(
                "Gemini insights failed (%s), using fallback message",
                gemini_exc,
            )
            insights = (
                f"**{zone_fmt['zone_name']}** — {zone_fmt['climate']}\n\n"
                f"**Soil:** {zone_fmt['soil']}\n\n"
                f"**Typical Crops:** {zone_fmt['crops']}\n\n"
                f"**Irrigation:** {zone_fmt['irrigation']}\n\n"
                f"**Challenges:** {zone_fmt['challenges']}\n\n"
                "⚠️ AI advice could not load right now — "
                "the zone info above is still helpful."
            )

        return LocationInsightsResponse(
            district=matched_district,
            zone_key=zone_key,
            zone_name=zone_data["name"],
            zone_icon=zone_data.get("icon", "📍"),
            climate=zone_data["climate"],
            soil=zone_data["soil"],
            crops=zone_data.get("crops", []),
            irrigation=zone_data["irrigation"],
            challenges=zone_data.get("challenges", []),
            temperature=temperature,
            humidity=humidity,
            weather_description=weather_desc,
            insights=insights,
        )

    except ValueError as exc:
        logger.warning("Location request validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        logger.error("Location service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected location error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
