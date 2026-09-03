"""
Yield prediction service — loads the trained ML model and generates
crop yield predictions, profit estimates, and weather-based risk warnings.

The model file (crop_yield_model.joblib) is produced by
``training/train_model.py`` and loaded once at service startup.
"""

import logging
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from backend.services.crop_economics import (
    CROP_ECONOMICS,
    CROP_WATER_PROFILE,
    SUPPORTED_CROPS,
)

logger = logging.getLogger(__name__)

# ── Model path ───────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_yield_model.joblib")


class YieldService:
    """Loads the trained model and runs predictions + profit calculations."""

    def __init__(self) -> None:
        """Load the saved model artifact from disk."""
        if not os.path.exists(_MODEL_PATH):
            raise RuntimeError(
                "Trained model not found. Run `python training/train_model.py` "
                f"first to generate the model at {_MODEL_PATH}."
            )

        # NOTE (security): joblib.load deserialises Python objects.
        # The model file is produced by our own training/train_model.py
        # script and stored locally — it is trusted provenance.  Do NOT
        # load model files from untrusted external sources.
        artifact = joblib.load(_MODEL_PATH)
        self._model = artifact["model"]
        self._feature_cols: List[str] = artifact["feature_cols"]
        self._model_name: str = artifact.get("model_name", "unknown")
        self._r2: float = artifact.get("metrics", {}).get("r2", 0.0)

        logger.info(
            "YieldService loaded: %s (R²=%.4f, features=%s)",
            self._model_name,
            self._r2,
            self._feature_cols,
        )

    # ── Public API ───────────────────────────────────────────────────

    def predict_yield(
        self,
        fertilizer: float,
        temp: float,
        n: float,
        p: float,
        k: float,
    ) -> float:
        """
        Predict crop yield using the trained ML model.

        Parameters
        ----------
        fertilizer, temp, n, p, k : float
            Input features matching the training dataset columns.

        Returns
        -------
        float
            Predicted yield (same unit as the training target).
        """
        # Pass a DataFrame so sklearn doesn't warn about missing feature names
        features = pd.DataFrame(
            [[fertilizer, temp, n, p, k]], columns=self._feature_cols
        )
        prediction = self._model.predict(features)[0]
        return round(float(prediction), 2)

    @staticmethod
    def calculate_profit(
        predicted_yield: float,
        crop: str,
        acres: float,
    ) -> dict:
        """
        Calculate estimated revenue, cost, and profit for a crop.

        Formula:
            Revenue  = Predicted Yield × acres × price_per_maud
            Cost     = cost_per_acre × acres
            Profit   = Revenue − Cost

        Parameters
        ----------
        predicted_yield : float
            Yield per acre from the ML model.
        crop : str
            Crop name (must exist in CROP_ECONOMICS).
        acres : float
            Farm size in acres.

        Returns
        -------
        dict with keys: revenue, cost, profit, profit_margin.
        """
        if crop not in CROP_ECONOMICS:
            raise ValueError(
                f"Unknown crop '{crop}'. "
                f"Supported: {', '.join(SUPPORTED_CROPS)}"
            )

        eco = CROP_ECONOMICS[crop]
        revenue = predicted_yield * acres * eco["price_per_maud"]
        cost = eco["cost_per_acre"] * acres
        profit = revenue - cost
        margin = (profit / revenue * 100) if revenue > 0 else 0.0

        return {
            "revenue": round(revenue, 0),
            "cost": round(cost, 0),
            "profit": round(profit, 0),
            "profit_margin": round(margin, 1),
        }

    @staticmethod
    def assess_weather_risk(
        crop: str,
        rain_chance_max: float,
        humidity: int,
        temperature: float,
    ) -> Optional[str]:
        """
        Compare the crop's water profile against actual weather conditions
        and return a risk warning if there's a mismatch.

        Parameters
        ----------
        crop : str
            Crop name.
        rain_chance_max : float
            Maximum precipitation probability from the 3-day forecast (0–1).
        humidity : int
            Current relative humidity (%).
        temperature : float
            Current temperature (°C).

        Returns
        -------
        str | None
            A human-readable risk warning, or None if conditions are fine.
        """
        if crop not in CROP_WATER_PROFILE:
            return None

        profile = CROP_WATER_PROFILE[crop]
        warnings: list[str] = []

        # High rain risk for crops that don't tolerate excess water
        if rain_chance_max >= 0.6 and profile["rain_tolerance"] == "low":
            warnings.append(
                f"⚠️ **Heavy rain expected** ({rain_chance_max:.0%} chance). "
                f"{crop} is sensitive to excess water — this could reduce "
                f"yield and quality. Consider delaying irrigation and ensure "
                f"proper field drainage."
            )

        # Drought risk for water-hungry crops
        if rain_chance_max <= 0.1 and profile["water_need"] == "high":
            warnings.append(
                f"⚠️ **Low rainfall forecast** (only {rain_chance_max:.0%} chance). "
                f"{crop} requires high water — drought conditions could "
                f"significantly reduce yield. Ensure adequate irrigation "
                f"is available."
            )

        # Extreme heat
        if temperature > 40 and crop in ("Wheat", "Maize"):
            warnings.append(
                f"⚠️ **Extreme heat** ({temperature}°C). "
                f"{crop} is stressed above 38°C — expect reduced "
                f"grain filling and lower yield."
            )

        # Very high humidity → fungal/disease risk
        if humidity > 85 and crop in ("Wheat", "Cotton", "Maize"):
            warnings.append(
                f"⚠️ **High humidity** ({humidity}%) increases "
                f"fungal disease risk for {crop}. Monitor for blight, "
                f"rust, or boll rot and apply fungicide if needed."
            )

        return "\n\n".join(warnings) if warnings else None

    @staticmethod
    def get_supported_crops() -> List[str]:
        """Return the list of crops with known economics data."""
        return SUPPORTED_CROPS
