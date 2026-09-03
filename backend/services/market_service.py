"""
Market price trend service — loads pre-computed seasonal patterns and
provides trend analysis for commodity price data.

The patterns artifact (market_patterns.joblib) is produced by
``training/train_market.py`` and loaded once at service startup.

Data Source:
    Pakistani mandi price data (2007–2022, 138 market towns) —
    per-crop daily wholesale prices combined from market-committee
    records.  Prices are PKR per maund (40 kg) for weight commodities;
    count commodities (kinnow/orange per 100 pieces, banana per dozen)
    keep their native unit, labelled clearly.
"""

import logging
import os

import joblib

logger = logging.getLogger(__name__)

# ── Model path ────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "market_patterns.joblib")

# Disclaimer shown in every response (subtle footnote-level text)
DISCLAIMER = (
    "Historical Pakistani mandi averages (2007–2022) — actual daily "
    "prices vary by market and season. Always confirm with your local "
    "mandi before making decisions."
)


class MarketService:
    """Loads market patterns and provides trend analysis for commodities."""

    def __init__(self) -> None:
        """Load the pre-computed market patterns artifact."""
        if not os.path.exists(_MODEL_PATH):
            raise RuntimeError(
                "Market patterns file not found. Run "
                "`python training/train_market.py` first to generate "
                f"the artifact at {_MODEL_PATH}."
            )

        # NOTE (security): joblib.load deserialises Python objects.
        # The artifact is produced by our own training script — trusted.
        artifact = joblib.load(_MODEL_PATH)
        self._commodities: dict = artifact.get("commodities", {})
        self._display_names: dict = artifact.get("display_names", {})
        self._date_range_start: str = artifact.get("date_range_start", "")
        self._date_range_end: str = artifact.get("date_range_end", "")

        logger.info(
            "MarketService loaded: %d commodities, date range %s to %s",
            len(self._commodities),
            self._date_range_start,
            self._date_range_end,
        )

    # ── Public API ────────────────────────────────────────────────────

    def get_commodity_list(self) -> list[dict]:
        """
        Return the list of available commodities with display names.

        Returns a list of dicts: [{"key": "Wheat", "name": "Wheat (Gandum)"}, ...]
        """
        return [
            {"key": key, "name": data.get("display_name", key)}
            for key, data in self._commodities.items()
        ]

    def get_trend(self, commodity_key: str) -> dict | None:
        """
        Return the full trend analysis for a commodity.

        Parameters
        ----------
        commodity_key : str
            Raw commodity name (must match a key in the patterns artifact).

        Returns
        -------
        dict or None
            Trend analysis data, or None if the commodity is not found.
        """
        data = self._commodities.get(commodity_key)
        if data is None:
            logger.warning("Commodity '%s' not found in patterns", commodity_key)
            return None

        # Build monthly stats list sorted by month number
        seasonal = data.get("seasonal_stats", {})
        monthly_stats = []
        for month_num in sorted(seasonal.keys()):
            stat = seasonal[month_num]
            monthly_stats.append({
                "month": stat["month"],
                "avg_price": stat["avg_price"],
                "min_price": stat["min_price"],
                "max_price": stat["max_price"],
            })

        return {
            "commodity": data.get("display_name", commodity_key),
            "trend": data.get("trend", "unknown"),
            "latest_price": data.get("latest_price", 0),
            "latest_date": data.get("latest_date", ""),
            "pct_change_3m": data.get("pct_change_3m", 0),
            "best_months": data.get("best_months", []),
            "worst_months": data.get("worst_months", []),
            "monthly_stats": monthly_stats,
            "chart_data": data.get("chart_data", []),
            "date_range_start": self._date_range_start,
            "date_range_end": self._date_range_end,
            "price_unit": data.get("price_unit", "Rs. / maund (40 kg)"),
        }

    @staticmethod
    def format_seasonal_summary(monthly_stats: list[dict]) -> str:
        """
        Format monthly stats into a human-readable seasonal summary.

        Returns a string like:
        "Prices historically highest in Jan, Feb (avg Rs. 2850-2870 per
        maund) and lowest in Apr, May (avg Rs. 2470-2480 per maund)."
        """
        if not monthly_stats:
            return "Insufficient seasonal data."

        # Find top 2 and bottom 2 months
        sorted_by_price = sorted(monthly_stats, key=lambda x: x["avg_price"], reverse=True)
        top = sorted_by_price[:2]
        bottom = sorted_by_price[-2:]

        top_str = ", ".join(
            f"{m['month']} (avg Rs. {m['avg_price']:,.0f}/maund)" for m in top
        )
        bottom_str = ", ".join(
            f"{m['month']} (avg Rs. {m['avg_price']:,.0f}/maund)" for m in bottom
        )
        return (
            f"Prices historically highest in {top_str} "
            f"and lowest in {bottom_str}."
        )
