"""
Market price trend analysis script — CropGenie HarvestAI
=========================================================

Processes the combined PAKISTANI mandi price dataset
(``data/pakistan_market_price_data.csv``, produced by
``training/combine_pakistan_market_data.py``) and computes seasonal
patterns and trend indicators for each commodity.

The dataset covers ~15 years (2007–2022) of daily mandi prices across
138 Pakistani market towns.  With this much history we can compute
genuine multi-year seasonality (which month of the year pays best):

  • Monthly national average prices (for charting, 2007–2022)
  • Month-of-year seasonal statistics (mean, std, min, max)
  • Best-selling months (highest average price across all years)
  • Trend direction — recent 12 months vs the previous 12 months
  • Recent 3-month price momentum

Prices are normalised to PKR per maund (40 kg) for weight-based
commodities.  Count-based commodities (kinnow/orange per 100 pieces,
banana per dozen) keep their natural unit, clearly labelled.

The output artifact is saved to ``models/market_patterns.joblib`` and
loaded at runtime by ``backend/services/market_service.py``.

Usage:
    python training/combine_pakistan_market_data.py   # run this FIRST
    python training/train_market.py
"""

import io
import logging
import os
import sys
from typing import Dict

# Windows consoles default to cp1252 — force UTF-8 (commodity names have no
# Urdu here, but display names do)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "pakistan_market_price_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PATH = os.path.join(MODEL_DIR, "market_patterns.joblib")

# Multiplier to convert raw prices to Rs/maund (None = keep native unit)
UNIT_TO_MAUND = {
    "maund": 1.0,
    "quintal": 0.40,   # 40 kg / 100 kg
    "100pcs": None,     # kinnow, orange, grapefruit, musambi, feutral
    "dozen": None,      # banana
}

# Friendly display names (English + Urdu hints)
DISPLAY_NAMES = {
    "paddy": "Paddy / Rice (Dhaan)",
    "mango": "Mango (Aam)",
    "gram": "Gram / Chana",
    "masoor": "Masoor Lentil (Daal Masoor)",
    "mash": "Mash (Daal Mash)",
    "moong": "Moong (Daal Moong)",
    "banana": "Banana (Kela)",
    "guava": "Guava (Amrood)",
    "orange": "Orange (Malta)",
    "kinnow": "Kinnow",
    "carrot": "Carrot (Gajar)",
    "garlic": "Garlic (Lehsan)",
    "green chilli": "Green Chilli (Hari Mirch)",
    "groundnut": "Groundnut (Moongphali)",
    "lemon": "Lemon (Limu)",
    "grapes": "Grapes (Angoor)",
    "grapefruit": "Grapefruit",
    "jaman": "Jaman (Jamun)",
    "jujube": "Jujube (Bair)",
    "jaggery": "Jaggery / Gur",
    "green onion": "Green Onion (Hari Pyaz)",
    "mustard": "Mustard (Sarson)",
    "ginger": "Ginger (Adrak)",
    "millet": "Millet (Bajra)",
    "melon": "Melon (Garma)",
    "mint": "Mint (Podina)",
    "apple": "Apple (Saib)",
    "apricot": "Apricot (Khubani)",
    "cabbage": "Cabbage (Bandh Gobhi)",
    "canola": "Canola",
    "capsicum": "Capsicum (Shimla Mirch)",
    "cauliflower": "Cauliflower (Phool Gobhi)",
    "cucumber": "Cucumber (Kheera)",
    "coriander": "Coriander (Dhaniya)",
    "cocoyam": "Cocoyam / Arvi",
    "coconut": "Coconut (Nariyal)",
    "brown sugar": "Brown Sugar (Shakkar)",
    "batho": "Batho (Bathua Greens)",
    "feutral": "Feutral Citrus",
    "musambi": "Musambi",
}

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ── Analysis Functions ────────────────────────────────────────────────

def analyse_commodity(df: pd.DataFrame, commodity: str) -> dict:
    """
    Compute monthly price patterns and trend for one commodity group.

    Parameters
    ----------
    df : pd.DataFrame
        Full combined dataset (already date-parsed).
    commodity : str
        Commodity group key (e.g. "paddy", "mango").

    Returns
    -------
    dict
        Structured analysis result (same shape as the previous Indian
        dataset artifact, so the market service keeps working).
    """
    cdf = df[df["commodity"] == commodity].copy()
    if cdf.empty:
        return {}

    # ── Unit handling — normalise weight-based prices to Rs/maund ──
    unit = cdf["unit"].mode().iloc[0]  # units are consistent per commodity
    multiplier = UNIT_TO_MAUND.get(unit)
    if multiplier is not None and multiplier != 1.0:
        cdf["price"] = cdf["price"] * multiplier
    if multiplier is not None:
        price_unit = "Rs. / maund (40 kg)"
    else:
        price_unit = {"100pcs": "Rs. / 100 pieces", "dozen": "Rs. / dozen"}[unit]

    # ── Monthly national averages → chart data ─────────────────────
    monthly = (
        cdf.groupby(cdf["date"].dt.to_period("M"))["price"]
        .mean()
        .reset_index()
    )
    monthly.columns = ["period", "avg_price"]
    monthly = monthly.sort_values("period")

    chart_data = [
        {"date": str(row["period"]), "price": round(row["avg_price"], 1)}
        for _, row in monthly.iterrows()
    ]

    # ── Month-of-year seasonal statistics (multi-year) ─────────────
    cdf["month"] = cdf["date"].dt.month
    seasonal = cdf.groupby("month")["price"].agg(
        mean="mean", std="std", min="min", max="max", count="count"
    ).reset_index()
    seasonal["std"] = seasonal["std"].fillna(0)

    seasonal_stats: Dict[int, dict] = {}
    for _, row in seasonal.iterrows():
        m = int(row["month"])
        seasonal_stats[m] = {
            "month": MONTH_NAMES[m],
            "avg_price": round(row["mean"], 1),
            "std": round(row["std"], 1),
            "min_price": round(row["min"], 1),
            "max_price": round(row["max"], 1),
            "data_points": int(row["count"]),
        }

    # ── Best / worst selling months (top / bottom 3 by avg) ───────
    sorted_months = sorted(
        seasonal_stats.values(), key=lambda x: x["avg_price"], reverse=True
    )
    best_months = [m["month"] for m in sorted_months[:3]]
    worst_months = [m["month"] for m in sorted_months[-3:]]

    # ── Trend: recent 12 months vs previous 12 months ─────────────
    if len(monthly) >= 24:
        y = monthly["avg_price"].values
        recent_avg = float(np.mean(y[-12:]))
        prior_avg = float(np.mean(y[-24:-12]))
        yearly_change = ((recent_avg - prior_avg) / prior_avg) * 100

        if yearly_change > 10:
            trend = "rising"
        elif yearly_change < -10:
            trend = "falling"
        else:
            trend = "stable"
    else:
        yearly_change = 0.0
        trend = "insufficient_data"

    # ── Recent 3-month momentum ───────────────────────────────────
    if len(monthly) >= 6:
        y = monthly["avg_price"].values
        pct_change = ((np.mean(y[-3:]) - np.mean(y[-6:-3])) / np.mean(y[-6:-3])) * 100
    else:
        pct_change = 0.0

    # ── Latest price — average of the last 3 months (more stable) ─
    last_periods = monthly.tail(3)
    latest_price = round(float(last_periods["avg_price"].mean()), 1)
    latest_date = str(monthly.iloc[-1]["period"])

    return {
        "display_name": DISPLAY_NAMES.get(commodity, commodity.title()),
        "raw_name": commodity,
        "total_records": len(cdf),
        "date_start": str(cdf["date"].min().date()),
        "date_end": str(cdf["date"].max().date()),
        "latest_price": latest_price,
        "latest_date": latest_date,
        "trend": trend,
        "yearly_change": round(yearly_change, 1),
        "pct_change_3m": round(pct_change, 1),
        "best_months": best_months,
        "worst_months": worst_months,
        "seasonal_stats": seasonal_stats,
        "chart_data": chart_data,
        "price_unit": price_unit,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """Run the full market price analysis pipeline."""
    if not os.path.exists(DATA_PATH):
        logger.error(
            "Combined dataset not found at %s.  Run "
            "`python training/combine_pakistan_market_data.py` first.",
            DATA_PATH,
        )
        return

    logger.info("Loading dataset from %s", DATA_PATH)
    # Read only the columns we need (the CSV is ~280 MB — this keeps
    # memory usage reasonable)
    df = pd.read_csv(
        DATA_PATH,
        usecols=["date", "market", "commodity", "crop", "price", "unit"],
        dtype={"market": "category", "commodity": "category",
               "crop": "category", "unit": "category"},
    )
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Dataset shape: %s", df.shape)
    logger.info(
        "Date range: %s to %s", df["date"].min().date(), df["date"].max().date()
    )

    commodities = sorted(df["commodity"].unique())
    logger.info("Commodities found: %d → %s", len(commodities), ", ".join(commodities))

    # Analyse every commodity in the dataset
    patterns: Dict[str, dict] = {}
    for commodity in commodities:
        result = analyse_commodity(df, commodity)
        if result:
            patterns[commodity] = result
            logger.info(
                "  %-18s: %7d records, trend=%-8s, latest=%s %s (%s)",
                result["display_name"],
                result["total_records"],
                result["trend"],
                result["latest_price"],
                result["price_unit"],
                result["latest_date"],
            )

    if not patterns:
        logger.error("No commodities found — aborting.")
        return

    # Save artifact (same structure the market service already expects)
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "commodities": patterns,
        "display_names": {
            k: v["display_name"] for k, v in patterns.items()
        },
        "date_range_start": str(df["date"].min().date()),
        "date_range_end": str(df["date"].max().date()),
        "source": "Pakistani mandi price data (2007–2022, 138 markets)",
        "note": (
            "Historical Pakistani mandi averages — actual daily prices "
            "vary by market and season. Always confirm with your local "
            "mandi before making decisions."
        ),
    }
    joblib.dump(artifact, OUTPUT_PATH)
    logger.info("Market patterns saved to %s", OUTPUT_PATH)
    logger.info(
        "Summary: %d commodities analysed, artifact size: %.1f KB",
        len(patterns),
        os.path.getsize(OUTPUT_PATH) / 1024,
    )


if __name__ == "__main__":
    main()
