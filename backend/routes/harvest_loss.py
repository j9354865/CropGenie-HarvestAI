"""
Harvest Loss Predictor route — POST /api/harvest-loss/predict
=============================================================

Tells a Pakistani farmer HOW MUCH yield and money they stand to lose
if they delay harvesting their crop, by combining:
  • Live 5-day weather forecast from OpenWeatherMap (rain risk)
  • Real market price trend from Pakistani mandi data (price direction)
  • Gemini AI reasoning (crop-specific harvest-timing knowledge)

The route returns three delay scenarios (Day 0 / Day 5 / Day 10)
with estimated yield % and loss %, plus an overall decision badge:
  🟢 HARVEST NOW  |  🟡 WAIT 2 DAYS  |  🔴 HARVEST BEFORE RAIN

Flow
----
1. Validate crop_name, location, growth_stage.
2. Fetch weather forecast (up to 5 days from OpenWeatherMap).
3. Fetch market price trend (from MarketService / market_patterns.joblib).
4. Build a structured prompt with real data and send to Gemini.
5. Parse Gemini's response — extract scenarios + decision badge.
6. Return structured JSON (frontend handles TTS separately, same as
   other pages in the app).
"""

import logging
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    HarvestLossRequest,
    HarvestLossResponse,
    HarvestLossScenario,
)
from backend.services.gemini_service import GeminiService
from backend.services.weather_service import WeatherService
from backend.services.market_service import MarketService

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy-initialised services ────────────────────────────────────────
_weather_service: WeatherService | None = None
_gemini_service: GeminiService | None = None
_market_service: MarketService | None = None


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


def _get_market_service() -> MarketService:
    """Lazy-init the market price trend service."""
    global _market_service
    if _market_service is None:
        _market_service = MarketService()
    return _market_service


# ── Helpers ───────────────────────────────────────────────────────────

# Language name mapping for the prompt (human-readable, not codes)
_LANG_NAME = {
    "en": "English",
    "ur": "Urdu",
    "ps": "Pashto",
}


def _format_weather_summary(forecast_days) -> str:
    """
    Turn a list of ForecastDay objects into a readable multi-day summary.

    Example:
        "Day 1 (2026-09-03): 32°C max, 10% rain | Day 2 ..."
    """
    parts = []
    for i, day in enumerate(forecast_days, 1):
        rain_pct = day.rain_chance * 100  # convert 0-1 to percentage
        parts.append(
            f"Day {i} ({day.date}): {day.temp_max:.0f}°C max, "
            f"{rain_pct:.0f}% rain chance, {day.description}"
        )
    return " | ".join(parts) if parts else "No forecast data available"


def _get_market_trend_summary(crop_name: str) -> tuple[str, bool]:
    """
    Look up the market price trend for a crop.

    Returns
    -------
    tuple[str, bool]
        (trend_summary_text, data_was_found)
    """
    try:
        market_svc = _get_market_service()

        # Try exact match first, then title-case
        trend = market_svc.get_trend(crop_name.strip())
        if trend is None:
            trend = market_svc.get_trend(crop_name.strip().title())
        if trend is None:
            # Try common farmer names → actual commodity keys in the dataset
            aliases = {
                "rice": "paddy", "dhaan": "paddy", "chawal": "paddy",
                "wheat": "wheat", "gandum": "wheat",
                "cotton": "cotton", "kapas": "cotton",
                "sugarcane": "sugarcane", "ganna": "sugarcane",
                "maize": "maize", "makai": "maize", "corn": "maize",
                "potato": "potato", "aloo": "potato",
                "tomato": "tomato", "tamatar": "tomato",
                "onion": "onion", "pyaaz": "onion",
                "mango": "mango", "aam": "mango",
                "banana": "banana", "kela": "banana",
                "guava": "guava", "amrood": "guava",
                "lemon": "lemon", "limu": "lemon",
                "orange": "orange", "malta": "orange",
                "mustard": "mustard", "sarson": "mustard",
                "canola": "canola",
                "gram": "gram", "chana": "gram",
                "millet": "millet", "bajra": "millet",
                "melon": "melon", "garma": "melon",
                "moong": "moong",
                "masoor": "masoor",
                "apple": "apple", "saib": "apple",
                "grapes": "grapes", "angoor": "grapes",
            }
            alias_key = crop_name.strip().lower()
            if alias_key in aliases:
                trend = market_svc.get_trend(aliases[alias_key])

        if trend is None:
            return "No price data available for this crop.", False

        # Build a human-readable trend summary
        trend_dir = trend.get("trend", "unknown")  # "rising", "falling", "stable"
        latest_price = trend.get("latest_price", 0)
        pct_change = trend.get("pct_change_3m", 0)
        price_unit = trend.get("price_unit", "Rs. / maund (40 kg)")
        latest_date = trend.get("latest_date", "")

        summary = (
            f"Current price: Rs. {latest_price:,.0f} {price_unit} "
            f"(as of {latest_date}). "
            f"3-month change: {pct_change:+.1f}%. "
            f"Overall trend: {trend_dir}."
        )
        return summary, True

    except Exception as exc:
        logger.warning("Market trend lookup failed: %s", exc)
        return "Market price data unavailable.", False


def _parse_harvest_response(raw_text: str) -> tuple[list[dict], str]:
    """
    Extract scenarios and decision badge from Gemini's response.

    Tries regex parsing for Day 0/5/10 yield percentages.
    Falls back to sensible defaults if parsing fails.

    Returns
    -------
    tuple[list[dict], str]
        (scenarios_list, decision_badge)
    """
    text_lower = raw_text.lower()

    # ── Parse decision badge ─────────────────────────────────────
    badge = "HARVEST NOW"  # safe default
    if "harvest before rain" in text_lower:
        badge = "HARVEST BEFORE RAIN"
    elif "wait 2 days" in text_lower or "wait two days" in text_lower:
        badge = "WAIT 2 DAYS"
    elif "harvest now" in text_lower:
        badge = "HARVEST NOW"

    # ── Parse scenario numbers ───────────────────────────────────
    # Look for patterns like "Day 0: 95% yield" or "Day 0 ... 95%"
    scenarios = []
    default_yields = {0: 95, 5: 85, 10: 70}

    for day in [0, 5, 10]:
        yield_pct = default_yields[day]  # fallback

        # Try multiple regex patterns
        patterns = [
            rf"[Dd]ay\s*{day}[:\s].*?(\d{{1,3}})\s*%",
            rf"[Dd]ay\s*{day}.*?yield.*?(\d{{1,3}})\s*%",
            rf"[Dd]ay\s*{day}.*?(\d{{1,3}})\s*%\s*yield",
        ]
        for pat in patterns:
            match = re.search(pat, raw_text, re.DOTALL)
            if match:
                parsed_val = int(match.group(1))
                # Sanity check: yield should be between 30 and 100
                if 30 <= parsed_val <= 100:
                    yield_pct = parsed_val
                    break

        loss_pct = max(0, 100 - yield_pct)
        scenarios.append({
            "delay_days": day,
            "expected_yield_pct": yield_pct,
            "loss_pct": loss_pct,
        })

    return scenarios, badge


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/harvest-loss/predict", response_model=HarvestLossResponse)
async def harvest_loss_predict(request: HarvestLossRequest):
    """
    Predict expected yield loss at different harvest-delay intervals.

    Combines live weather forecast, market price trend, and Gemini AI
    reasoning to tell the farmer:
      - What happens if they harvest now vs. 5 vs. 10 days later
      - An overall decision: HARVEST NOW / WAIT 2 DAYS / HARVEST BEFORE RAIN
    """
    try:
        crop_name = request.crop_name.strip()
        location = request.location.strip()
        language = request.language.strip() or "en"

        if not crop_name:
            raise HTTPException(status_code=400, detail="Crop name is required.")
        if not location:
            raise HTTPException(status_code=400, detail="Location is required.")

        # ── Step 1: Fetch weather forecast (5 days) ─────────────
        weather_text = "Weather forecast unavailable (API error)."
        weather_used = False
        resolved_city = location

        try:
            weather_svc = _get_weather_service()
            current = weather_svc.get_current_weather(city=location)
            forecast = weather_svc.get_forecast(city=location, days=5)
            resolved_city = current.city

            weather_text = (
                f"Current: {current.temperature:.1f}°C, "
                f"{current.humidity}% humidity, {current.description}. "
                f"5-day forecast: {_format_weather_summary(forecast)}"
            )
            weather_used = True

            logger.info(
                "Harvest-loss weather for %s: %s", location, weather_text,
            )
        except Exception as weather_exc:
            logger.warning(
                "Harvest-loss weather fetch failed for %s: %s",
                location, weather_exc,
            )

        # ── Step 2: Fetch market price trend ────────────────────
        market_text, market_used = _get_market_trend_summary(crop_name)

        logger.info(
            "Harvest-loss market for %s: %s (found=%s)",
            crop_name, market_text, market_used,
        )

        # ── Step 3: Build the Gemini prompt ─────────────────────
        lang_name = _LANG_NAME.get(language, "English")
        planned_text = (
            f"Planned harvest date: {request.planned_harvest_date}"
            if request.planned_harvest_date
            else "No planned date (today = Day 0)"
        )

        prompt = (
            f"Crop: {crop_name}\n"
            f"Growth stage: {request.growth_stage}\n"
            f"Location: {resolved_city}\n"
            f"{planned_text}\n"
            f"Weather forecast (next 5 days): {weather_text}\n"
            f"Current market price trend: {market_text}\n\n"
            f"Estimate the expected yield and financial impact of "
            f"delaying harvest, for these three scenarios:\n"
            f"1. Harvest now (Day 0)\n"
            f"2. Harvest 5 days later\n"
            f"3. Harvest 10 days later\n\n"
            f"For each scenario, estimate: expected yield (as a percentage "
            f"of maximum) and loss percentage compared to harvesting at the "
            f"best time.\n\n"
            f"Then give an overall Harvest Decision Score considering: "
            f"weather risk (especially upcoming rain), crop maturity/growth "
            f"stage, and market price trend.\n\n"
            f"Give a final clear recommendation as one of exactly these "
            f"three labels: 'HARVEST NOW', 'WAIT 2 DAYS', or "
            f"'HARVEST BEFORE RAIN'.\n\n"
            f"IMPORTANT LANGUAGE INSTRUCTIONS: Respond ONLY in {lang_name}. "
            f"Write in simple, natural, everyday {lang_name} that an average "
            f"farmer with basic literacy can easily understand. Use short "
            f"sentences. If a common English technical term is more "
            f"recognizable to farmers than its formal translation, keep it "
            f"in English within the sentence.\n\n"
            f"Keep the entire response concise — under 150 words — "
            f"structured with clear labels for each scenario and the "
            f"final recommendation."
        )

        # ── Step 4: Ask Gemini for harvest-loss prediction ──────
        try:
            gemini_svc = _get_gemini_service()
            raw_advice = gemini_svc.generate_harvest_loss_advice(
                prompt, language=language,
            )
        except Exception as gemini_exc:
            # Gemini failed — return a friendly fallback
            logger.warning(
                "Gemini harvest-loss advice failed (%s), using fallback",
                gemini_exc,
            )
            # Fallback message in the requested language if possible
            fallback_messages = {
                "en": (
                    "Unable to generate harvest prediction right now. "
                    "Please try again in a moment."
                ),
                "ur": (
                    "Abhi fasal ki katai ka andaza nahi lagaya ja saka. "
                    "Barah-e-karam dobara koshish karen."
                ),
                "ps": (
                    "Da sungay warkaw na shee. "
                    "Meharbani wokray biya hata kawka."
                ),
            }
            raw_advice = fallback_messages.get(language, fallback_messages["en"])

        # ── Step 5: Parse scenarios + badge from Gemini text ────
        scenarios_data, badge = _parse_harvest_response(raw_advice)

        # Build the scenario list using our Pydantic model
        scenarios = [
            HarvestLossScenario(
                delay_days=s["delay_days"],
                expected_yield_pct=s["expected_yield_pct"],
                loss_pct=s["loss_pct"],
            )
            for s in scenarios_data
        ]

        # ── Step 6: Return structured response ─────────────────
        return HarvestLossResponse(
            crop=crop_name,
            location=resolved_city,
            language=language,
            scenarios=scenarios,
            decision_badge=badge,
            advice_text=raw_advice,
            weather_used=weather_used,
            market_used=market_used,
        )

    except HTTPException:
        raise  # re-raise validation errors unchanged

    except Exception as exc:
        # Catch-all — never crash
        logger.error("Unexpected harvest-loss error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
