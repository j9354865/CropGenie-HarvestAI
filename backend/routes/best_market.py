"""
Best Selling Market route — /api/best-market/recommend
=======================================================

Tells a Pakistani farmer WHICH mandi (market town) pays the highest
price for their crop, based on the combined Pakistani mandi price
dataset (2007–2022, 138 markets).

Flow:
  1. Look up the crop's recent per-market prices (BestMarketService)
  2. Rank markets by average price, converted to Rs/maund + Rs/bori
  3. Ask Gemini for a recommendation that factors in a rough
     intercity transport-cost estimate (PKR) — is travelling to the
     higher-price mandi actually worth it?
  4. Return the structured market list + AI advice
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    BestMarketRequest,
    BestMarketResponse,
    MarketPrice,
)
from backend.services.best_market_service import BestMarketService
from backend.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised services
_best_market_service: BestMarketService | None = None
_gemini_service: GeminiService | None = None


def _get_best_market_service() -> BestMarketService:
    """Lazy-init the best-market service."""
    global _best_market_service
    if _best_market_service is None:
        _best_market_service = BestMarketService()
    return _best_market_service


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


@router.post("/best-market/recommend", response_model=BestMarketResponse)
async def best_market_recommend(request: BestMarketRequest):
    """
    Recommend the best mandi to sell a crop, with transport-cost advice.
    """
    try:
        market_svc = _get_best_market_service()
        gemini_svc = _get_gemini_service()

        crop_query = request.crop.strip()
        location = request.location.strip() if request.location else ""

        # ── Step 1: Get ranked market recommendations ──────────────
        result = market_svc.recommend(crop_query, location=location)
        if result is None:
            # Give the farmer a helpful list of what we DO have
            available = [
                c["name"] for c in market_svc.get_commodity_list()
            ]
            raise ValueError(
                f"No price data found for '{crop_query}'. "
                f"Try one of these crops: {', '.join(available[:40])}"
            )

        # ── Step 2: Build the markets summary for Gemini ───────────
        lines = []
        for m in result["markets"]:
            line = f"• {m['market']}: Rs. {m['avg_price']:,.0f}"
            if m.get("price_per_bori"):
                line += f" per maund (≈ Rs. {m['price_per_bori']:,.0f} per bori)"
            else:
                line += f" {m['unit_label']}"
            lines.append(line)
        markets_summary = "\n".join(lines)

        # ── Step 3: Generate AI advice with transport estimate ─────
        best = result["markets"][0] if result["markets"] else None
        try:
            advice = gemini_svc.generate_best_market_advice(
                crop=result["crop"],
                markets_summary=markets_summary,
                home_market=result.get("home_market") or "",
                home_price=result.get("home_price") or 0,
                best_market=best["market"] if best else "",
                best_price=best["avg_price"] if best else 0,
                unit_label=result["unit_label"],
                best_gain=result.get("best_gain_vs_home"),
                best_gain_pct=result.get("best_gain_pct"),
                location=location,
                language=request.language,
            )
        except Exception as gemini_exc:
            # Gemini failed — fall back to a plain data-driven summary
            logger.warning(
                "Gemini best-market advice failed (%s), using fallback", gemini_exc
            )
            fallback_lines = [
                f"**{result['crop']}** — top markets by recent price:",
                *lines,
                "",
                f"National average: Rs. {result['national_avg']:,.0f} "
                f"{result['unit_label']}",
            ]
            if result.get("home_market"):
                fallback_lines.append(
                    f"Your nearest market ({result['home_market']}): "
                    f"Rs. {result['home_price']:,.0f} {result['unit_label']}"
                )
                if result.get("best_gain_vs_home") is not None:
                    fallback_lines.append(
                        f"Best market pays {result['best_gain_vs_home']:+,.0f} "
                        f"Rs/maund more ({result['best_gain_pct']:+.1f}%) — "
                        f"remember to subtract transport costs before deciding."
                    )
            fallback_lines.append(
                "\n⚠️ AI advice could not load right now — check the prices "
                "above and your transport cost before going."
            )
            advice = "\n".join(fallback_lines)

        # ── Step 4: Build the response ─────────────────────────────
        market_prices = [
            MarketPrice(
                market=m["market"],
                avg_price=m["avg_price"],
                price_per_bori=m.get("price_per_bori"),
                unit_label=m.get("unit_label", result["unit_label"]),
                min_price=m["min_price"],
                max_price=m["max_price"],
                records=m["records"],
                last_date=m["last_date"],
            )
            for m in result["markets"]
        ]

        return BestMarketResponse(
            crop=result["crop"],
            commodity=result["commodity"],
            unit_label=result["unit_label"],
            bag_weight_kg=result.get("bag_weight_kg"),
            markets=market_prices,
            national_avg=result["national_avg"],
            home_market=result.get("home_market"),
            home_price=result.get("home_price"),
            best_gain_vs_home=result.get("best_gain_vs_home"),
            best_gain_pct=result.get("best_gain_pct"),
            last_date=result["last_date"],
            advice=advice,
        )

    except ValueError as exc:
        logger.warning("Best-market validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        logger.error("Best-market service error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected best-market error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )


@router.get("/best-market/crops")
async def best_market_crops():
    """Return the list of searchable crop varieties for the dropdown."""
    market_svc = _get_best_market_service()
    return {"crops": market_svc.get_commodity_list()}
