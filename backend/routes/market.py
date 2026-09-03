"""
Market price trend route — /api/market-trend

Loads pre-computed seasonal price patterns for agricultural commodities,
provides trend analysis and chart data, then asks Gemini for
farmer-friendly selling advice.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    ChartPoint,
    MarketTrendRequest,
    MarketTrendResponse,
    MonthlyStat,
)
from backend.services.gemini_service import GeminiService
from backend.services.location_service import LocationService
from backend.services.market_service import DISCLAIMER, MarketService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised services
_market_service: MarketService | None = None
_gemini_service: GeminiService | None = None
_location_service: LocationService | None = None


def _get_market_service() -> MarketService:
    """Lazy-init the market service."""
    global _market_service
    if _market_service is None:
        _market_service = MarketService()
    return _market_service


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


def _get_location_service() -> LocationService:
    """Lazy-init the location service."""
    global _location_service
    if _location_service is None:
        _location_service = LocationService()
    return _location_service


@router.post("/market-trend", response_model=MarketTrendResponse)
async def market_trend(request: MarketTrendRequest):
    """
    Analyse price trends for a commodity and generate selling advice.

    1. Load pre-computed seasonal patterns for the commodity
    2. Build trend summary + chart data
    3. Pass trend data to Gemini for farmer-friendly advice
    4. Return structured response with chart data, stats, and AI advice
    """
    try:
        market_svc = _get_market_service()
        gemini_svc = _get_gemini_service()

        commodity_key = request.commodity.strip()
        city = request.city.strip() if request.city else ""

        # ── Step 1: Get trend data ──────────────────────────────────
        trend_data = market_svc.get_trend(commodity_key)
        if trend_data is None:
            available = [c["name"] for c in market_svc.get_commodity_list()]
            raise ValueError(
                f"Commodity '{commodity_key}' not found. "
                f"Available: {', '.join(available)}"
            )

        # ── Step 2: Build monthly stats + chart for response ────────
        monthly_stats = [
            MonthlyStat(
                month=s["month"],
                avg_price=s["avg_price"],
                min_price=s["min_price"],
                max_price=s["max_price"],
            )
            for s in trend_data["monthly_stats"]
        ]

        chart_data = [
            ChartPoint(date=p["date"], price=p["price"])
            for p in trend_data["chart_data"]
        ]

        # ── Step 2b: Resolve location context (optional) ─────────
        matched_city = None
        zone_name = None
        zone_icon = None
        location_context = ""

        if city:
            loc_svc = _get_location_service()
            zone_result = loc_svc.resolve_zone(city)
            if zone_result:
                district, zone_key, zone_data = zone_result
                matched_city = LocationService._find_matched_district(
                    district, zone_key
                )
                zone_name = zone_data.get("name", "")
                zone_icon = zone_data.get("icon", "")
                zone_info = LocationService.format_zone_for_gemini(zone_data)
                location_context = (
                    f"City: {matched_city}\n"
                    f"Zone: {zone_info['zone_name']}\n"
                    f"Climate: {zone_info['climate']}\n"
                    f"Soil: {zone_info['soil']}\n"
                    f"Major Crops: {zone_info['crops']}\n"
                    f"Irrigation: {zone_info['irrigation']}\n"
                    f"Challenges: {zone_info['challenges']}"
                )
            else:
                # City provided but not found — still pass the city name
                matched_city = city
                location_context = f"City: {city} (zone not identified)"

        # ── Step 3: Generate AI advice ──────────────────────────────
        seasonal_summary = market_svc.format_seasonal_summary(
            trend_data["monthly_stats"]
        )

        try:
            advice = gemini_svc.generate_market_advice(
                commodity=trend_data["commodity"],
                trend=trend_data["trend"],
                latest_price=trend_data["latest_price"],
                pct_change=trend_data["pct_change_3m"],
                best_months=", ".join(trend_data["best_months"]),
                worst_months=", ".join(trend_data["worst_months"]),
                seasonal_summary=seasonal_summary,
                location_context=location_context,
                unit_label=trend_data.get("price_unit", "Rs. / maund (40 kg)"),
                language=request.language,
            )
        except Exception as gemini_exc:
            logger.warning(
                "Gemini market advice failed (%s), using fallback", gemini_exc
            )
            advice = (
                f"**{trend_data['commodity']}** — Trend: **{trend_data['trend']}**\n\n"
                f"{seasonal_summary}\n\n"
                f"Best months to sell: {', '.join(trend_data['best_months'])}\n"
                f"Lowest price months: {', '.join(trend_data['worst_months'])}\n\n"
                f"⚠️ AI advice could not load right now — "
                f"use the price data above as a guide."
            )

        return MarketTrendResponse(
            commodity=trend_data["commodity"],
            trend=trend_data["trend"],
            latest_price=trend_data["latest_price"],
            latest_date=trend_data["latest_date"],
            price_unit=trend_data.get("price_unit", "Rs. / maund (40 kg)"),
            pct_change_3m=trend_data["pct_change_3m"],
            best_months=trend_data["best_months"],
            worst_months=trend_data["worst_months"],
            seasonal_stats=monthly_stats,
            chart_data=chart_data,
            date_range_start=trend_data["date_range_start"],
            date_range_end=trend_data["date_range_end"],
            advice=advice,
            matched_city=matched_city,
            zone_name=zone_name,
            zone_icon=zone_icon,
            disclaimer=DISCLAIMER,
        )

    except ValueError as exc:
        logger.warning("Market trend validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        logger.error("Market service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected market error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )


@router.get("/market-commodities")
async def market_commodities():
    """Return the list of available commodities for the dropdown."""
    market_svc = _get_market_service()
    return {"commodities": market_svc.get_commodity_list()}
