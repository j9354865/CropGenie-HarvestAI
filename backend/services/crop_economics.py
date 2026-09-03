"""
Crop economics reference data — market prices, farming costs, and water
requirements for common crops.

IMPORTANT: These figures are approximate placeholder estimates based on
general Pakistan / South-Asia agriculture data (2024–2025).  They should
be periodically updated with current local market rates for production use.

All prices are in PKR (Pakistani Rupees) per unit:
    • yield_unit : "maunds per acre" (1 maund ≈ 37.324 kg)
    • price      : PKR per maund
    • cost       : PKR per acre (seed + fertilizer + labour + irrigation + misc.)

Yield-scale note
----------------
The ML model (crop_yield_model.joblib) was trained on a crop-agnostic
agronomic dataset whose target column spans ~5–12 maunds/acre (P50 ≈ 8.5).
The model therefore produces the *same* numeric scale for every crop;
it encodes relative productivity (nutrient levels, temperature, fertilizer)
not absolute real-world tonnage per crop variety.

Cost calibration rule
---------------------
cost_per_acre is set to 72 % of (P50_yield × price_per_maud) so that:
    • At median yield (≈ 8.5 maunds) : profit margin ≈ +27–29 %   (good)
    • At P25 yield   (≈ 6.9 maunds) : profit margin ≈ +10–13 %   (marginal)
    • At P10 yield   (≈ 5.9 maunds) : profit margin ≈  −2– +2 %  (breakeven / small loss)
This ensures realistic, crop-consistent profit signals across the full
yield distribution instead of uniformly negative or unrealistic numbers.
"""

from typing import Dict


# ── Market Economics ─────────────────────────────────────────────────
#
# price_per_maud  : average market selling price (PKR per maund)
# cost_per_acre   : total farming cost per acre (PKR)
# typical_yield   : average expected yield (maunds per acre) — used as
#                   a sanity check; the ML model provides the actual
#                   prediction.
#
# NOTE: Update these figures periodically for accuracy.
#
# Calibration: cost_per_acre = round(P50_yield × price_per_maud × 0.72, -3)
#   P50_yield ≈ 8.50 maunds/acre from the training dataset.
#   This gives ~27-29 % margin at median conditions for all crops.

CROP_ECONOMICS: Dict[str, dict] = {
    "Wheat": {
        "price_per_maud": 4200,    # PKR/maund (≈ official 2024-25 support price)
        "cost_per_acre": 26000,    # PKR/acre — calibrated to P50 yield
        "typical_yield": 8,        # maunds/acre — model P50
    },
    "Rice": {
        "price_per_maud": 5500,    # PKR/maund (basmati / common, farm gate)
        "cost_per_acre": 34000,    # PKR/acre — calibrated to P50 yield
        "typical_yield": 8,
    },
    "Cotton": {
        "price_per_maud": 7000,    # PKR/maund (seed cotton)
        "cost_per_acre": 43000,    # PKR/acre — calibrated to P50 yield
        "typical_yield": 8,
    },
    "Maize": {
        "price_per_maud": 3800,    # PKR/maund
        "cost_per_acre": 23000,    # PKR/acre — calibrated to P50 yield
        "typical_yield": 8,
    },
    "Sugarcane": {
        "price_per_maud": 800,     # PKR/maund (mill gate price)
        "cost_per_acre": 5000,     # PKR/acre — calibrated to P50 yield
        "typical_yield": 8,        # model yield-index unit, not real-world maunds
    },
}

# List of supported crops for the frontend dropdown
SUPPORTED_CROPS = sorted(CROP_ECONOMICS.keys())


# ── Water Requirement Profiles ───────────────────────────────────────
#
# water_need      : qualitative — "low", "moderate", "high"
# rain_tolerance  : how well the crop handles heavy rain
#                   ("low" = damaged by excess water)
# drought_risk    : how badly the crop suffers from water shortage
#                   ("high" = significant yield loss in drought)

CROP_WATER_PROFILE: Dict[str, dict] = {
    "Wheat": {
        "water_need": "moderate",
        "rain_tolerance": "low",
        "drought_risk": "moderate",
    },
    "Rice": {
        "water_need": "high",
        "rain_tolerance": "high",
        "drought_risk": "high",
    },
    "Cotton": {
        "water_need": "low-moderate",
        "rain_tolerance": "low",
        "drought_risk": "moderate",
    },
    "Maize": {
        "water_need": "moderate",
        "rain_tolerance": "moderate",
        "drought_risk": "moderate",
    },
    "Sugarcane": {
        "water_need": "high",
        "rain_tolerance": "moderate",
        "drought_risk": "high",
    },
}
