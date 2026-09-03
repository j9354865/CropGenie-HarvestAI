"""
Best Market recommendation service — CropGenie HarvestAI
=========================================================

Loads the pre-computed Pakistani mandi price summary
(``models/pakistan_market_best.joblib``, produced by
``training/combine_pakistan_market_data.py``) and recommends the best
mandi/market to sell a given crop, comparing prices across Pakistani
market towns.

Farmer-friendly units
---------------------
All prices are reported in:
  • **Rs. per maund** (1 maund = 40 kg — the traditional Pakistani
    mandi unit farmers actually use), and
  • **Rs. per bori** (jute bag) using a configurable bag weight
    (``BAG_WEIGHT_KG``, default 100 kg — typical for wheat/rice/pulses).

Raw data units vary per crop (maund / quintal / 100 pieces / dozen) and
are converted using the per-crop multiplier stored in the artifact.
Count-based crops (kinnow per 100 pcs, banana per dozen) keep their
native unit — no fake conversion to maund is attempted.

Usage flow (from the route):
    service = BestMarketService()
    service.find_crop("paddy")            → matching crop varieties
    service.recommend("paddy", "Okara")   → ranked market recommendations
"""

import logging
import os
from typing import Optional

import joblib

logger = logging.getLogger(__name__)

# ── Model path ────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "models", "pakistan_market_best.joblib")

# ── Farmer-friendly unit constants ───────────────────────────────────

# 1 maund = 40 kg (traditional Pakistani mandi weight unit)
MAUND_KG = 40

# Assumed bori (jute bag) weight for the "price per bag" display.
# 100 kg is typical for wheat, rice, and pulses in Pakistan.
# → change this one constant if you want a different bag size.
BAG_WEIGHT_KG = 100

# ── Search helpers ───────────────────────────────────────────────────

# Common farmer inputs → commodity/crop aliases
_SEARCH_ALIASES = {
    "rice": "paddy",
    "dhaan": "paddy",
    "chana": "gram",
    "channa": "gram",
    "daal": None,   # ambiguous — matched generically below
    "dal": None,
    "kela": "banana",
    "amrood": "guava",
    "aam": "mango",
    "saib": "apple",
    "gajar": "carrot",
    "lehsan": "garlic",
    "adrak": "ginger",
    "sarson": "mustard",
    "bajra": "millet",
    "angoor": "grapes",
    "limu": "lemon",
    "mirch": "green chilli",
    "hari mirch": "green chilli",
    "moongphali": "groundnut",
    "pyaz": "green onion",
    "gur": "jaggery",
    "khubani": "apricot",
    "makai": None,
    "kapas": None,
}


class BestMarketService:
    """Recommends the best Pakistani mandi to sell a crop."""

    def __init__(self) -> None:
        """Load the pre-computed market summary artifact."""
        if not os.path.exists(_ARTIFACT_PATH):
            raise RuntimeError(
                "Pakistani market summary not found. Run "
                "`python training/combine_pakistan_market_data.py` first "
                f"to generate the artifact at {_ARTIFACT_PATH}."
            )

        # NOTE (security): joblib.load deserialises Python objects.
        # The artifact is produced by our own script — trusted.
        artifact = joblib.load(_ARTIFACT_PATH)
        self._crops: dict = artifact.get("crops", {})

        logger.info(
            "BestMarketService loaded: %d crop varieties across %d commodities",
            len(self._crops),
            len({c.get("commodity") for c in self._crops.values()}),
        )

    # ── Public API ──────────────────────────────────────────────────

    def get_commodity_list(self) -> list[dict]:
        """
        Return all searchable crop varieties with display names.

        Returns [{"key": "Paddy Basmati", "name": "Paddy Basmati"}, ...]
        sorted by commodity then crop name.
        """
        items = []
        seen = set()
        for crop_key, data in self._crops.items():
            commodity = data.get("commodity", "")
            if (commodity, crop_key) in seen:
                continue
            seen.add((commodity, crop_key))
            items.append({"key": crop_key, "name": crop_key})
        return sorted(items, key=lambda x: (x["name"]))

    def find_crop(self, query: str) -> list[str]:
        """
        Find crop varieties matching a farmer's free-text query.

        Matches against the crop name and the commodity group
        (e.g. "rice" → paddy varieties, "basmati" → Paddy Basmati).
        Returns a list of crop keys (may be empty).
        """
        q = (query or "").strip().lower()
        if not q:
            return []

        # Alias expansion (e.g. "rice" → "paddy")
        alias = _SEARCH_ALIASES.get(q)
        if alias is None and q in _SEARCH_ALIASES:
            return []  # explicitly ambiguous alias
        search_terms = [q]
        if alias:
            search_terms.append(alias)

        matches = []
        for crop_key, data in self._crops.items():
            crop_lower = crop_key.lower()
            commodity = data.get("commodity", "").lower()
            if any(
                t in crop_lower or t in commodity or crop_lower in t
                for t in search_terms
            ):
                matches.append(crop_key)
        return sorted(matches)

    def recommend(
        self,
        crop_query: str,
        location: str = "",
        top_n: int = 5,
    ) -> Optional[dict]:
        """
        Recommend the best markets to sell a crop.

        Parameters
        ----------
        crop_query : str
            Crop name or commodity (e.g. "paddy", "basmati", "mango").
        location : str
            The farmer's city/district (optional — used to compute the
            price gap vs their nearest market and for the transport-
        top_n : int
            How many markets to rank.

        Returns
        -------
        dict or None
            {
                "crop": "Paddy Basmati",
                "commodity": "paddy",
                "unit_label": "Rs. / maund (40 kg)",
                "bag_weight_kg": 100,
                "markets": [
                    {
                        "market": "Okara",
                        "avg_price": 6250,          # Rs/maund
                        "price_per_bori": 15625,     # Rs/100kg bag
                        "min_price": ..., "max_price": ...,
                        "records": ..., "last_date": "...",
                    }, ...
                ],
                "national_avg": 5911,
                "home_market": "Okara" | None,
                "home_price": 6100 | None,
                "best_gain_vs_home": 150,            # Rs/maund
                "best_gain_pct": 2.5,
            }
            or None if the crop is not found.
        """
        # ── Resolve the crop ────────────────────────────────────────
        matches = self.find_crop(crop_query)
        if not matches:
            return None

        # Prefer an exact-ish match, else the first variety
        q = crop_query.strip().lower()
        crop_key = next(
            (m for m in matches if m.lower() == q or q in m.lower()),
            matches[0],
        )
        data = self._crops[crop_key]
        unit = data["unit"]
        mult = data.get("per_maund_multiplier")  # None for count-based

        # ── Rank markets by recent average price ────────────────────
        markets = sorted(
            data["markets"].items(),
            key=lambda kv: kv[1]["avg_price"],
            reverse=True,
        )[:top_n]

        market_rows = []
        for market_name, mdata in markets:
            raw_avg = mdata["avg_price"]
            if mult is not None:
                avg_maund = round(raw_avg * mult)
                per_bori = round(avg_maund * (BAG_WEIGHT_KG / MAUND_KG))
            else:
                avg_maund = None
                per_bori = None
            market_rows.append({
                "market": market_name,
                "avg_price": avg_maund if mult is not None else raw_avg,
                "price_per_bori": per_bori,
                "unit": unit,
                "unit_label": (
                    "Rs. / maund" if mult is not None
                    else data.get("unit_label", "")
                ),
                "min_price": (
                    round(mdata["min_price"] * mult) if mult is not None
                    else mdata["min_price"]
                ),
                "max_price": (
                    round(mdata["max_price"] * mult) if mult is not None
                    else mdata["max_price"]
                ),
                "records": mdata["records"],
                "last_date": mdata["last_date"],
            })

        # ── National average ────────────────────────────────────────
        raw_nat = data.get("national_avg_price", 0)
        national_avg = round(raw_nat * mult) if mult is not None else raw_nat

        # ── Home-market lookup (case-insensitive substring) ────────
        home_market = None
        home_price = None
        if location:
            loc = location.strip().lower()
            for market_name, mdata in data["markets"].items():
                if loc in market_name.lower() or market_name.lower() in loc:
                    home_market = market_name
                    home_price = (
                        round(mdata["avg_price"] * mult) if mult is not None
                        else mdata["avg_price"]
                    )
                    break

        # ── Best gain vs home market ───────────────────────────────
        best_gain = None
        best_gain_pct = None
        if home_price and market_rows:
            top_price = market_rows[0]["avg_price"]
            if home_price > 0:
                best_gain = top_price - home_price
                best_gain_pct = round(
                    (top_price - home_price) / home_price * 100, 1
                )

        return {
            "crop": crop_key,
            "commodity": data.get("commodity", ""),
            "unit": unit,
            "unit_label": (
                "Rs. / maund (40 kg)" if mult is not None
                else data.get("unit_label", "")
            ),
            "is_weight_based": mult is not None,
            "bag_weight_kg": BAG_WEIGHT_KG if mult is not None else None,
            "markets": market_rows,
            "national_avg": national_avg,
            "home_market": home_market,
            "home_price": home_price,
            "best_gain_vs_home": best_gain,
            "best_gain_pct": best_gain_pct,
            "last_date": data.get("last_date", ""),
            "total_records": data.get("total_records", 0),
        }
