"""
Crop recommendation service — loads the trained ML classifier and
generates crop predictions with confidence scores.

The model file (crop_recommendation_model.joblib) is produced by
``training/train_crop_recommendation.py`` and loaded once at service startup.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Model path ───────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_recommendation_model.joblib")


class CropRecommendationService:
    """Loads the trained crop classifier and runs predictions."""

    def __init__(self) -> None:
        """Load the saved model artifact from disk."""
        if not os.path.exists(_MODEL_PATH):
            raise RuntimeError(
                "Trained crop recommendation model not found. Run "
                "`python training/train_crop_recommendation.py` first to generate "
                f"the model at {_MODEL_PATH}."
            )

        # NOTE (security): joblib.load deserialises Python objects.
        # The model file is produced by our own training script and stored
        # locally — it is trusted provenance.  Do NOT load model files from
        # untrusted external sources.
        artifact = joblib.load(_MODEL_PATH)
        self._model = artifact["model"]
        self._feature_cols: list[str] = artifact["feature_cols"]
        self._label_encoder = artifact["label_encoder"]
        self._model_name: str = artifact.get("model_name", "unknown")
        self._accuracy: float = artifact.get("metrics", {}).get("accuracy", 0.0)
        self._crop_names: list[str] = artifact.get("crop_names", [])

        logger.info(
            "CropRecommendationService loaded: %s (accuracy=%.4f, features=%s)",
            self._model_name,
            self._accuracy,
            self._feature_cols,
        )

    # ── Public API ───────────────────────────────────────────────────

    def predict(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
    ) -> dict:
        """
        Predict the best crop to grow given soil and climate conditions.

        Parameters
        ----------
        nitrogen, phosphorus, potassium : float
            Soil nutrient levels (N, P, K) in kg/ha.
        temperature : float
            Average temperature in °C.
        humidity : float
            Relative humidity percentage (0–100).
        ph : float
            Soil pH level (0–14).
        rainfall : float
            Annual rainfall in mm.

        Returns
        -------
        dict with keys:
            recommended_crop : str — top predicted crop
            confidence : float — model confidence for top crop (0–1)
            top_3 : list[dict] — top 3 crops with confidence scores
        """
        # Build feature row — must match feature_cols order from training
        row = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }
        features = pd.DataFrame([row], columns=self._feature_cols)

        # Predict class probabilities for ALL crops
        probas = self._model.predict_proba(features)[0]

        # Get the indices sorted by probability (highest first)
        sorted_indices = np.argsort(probas)[::-1]

        # Top prediction
        top_idx = sorted_indices[0]
        recommended_crop = self._label_encoder.inverse_transform([top_idx])[0]
        confidence = float(probas[top_idx])

        # Top 3 alternatives
        top_3 = []
        for i in range(min(3, len(sorted_indices))):
            idx = sorted_indices[i]
            crop_name = self._label_encoder.inverse_transform([idx])[0]
            top_3.append({
                "crop": crop_name,
                "confidence": float(probas[idx]),
            })

        return {
            "recommended_crop": recommended_crop,
            "confidence": confidence,
            "top_3": top_3,
        }

    def get_supported_crops(self) -> list[str]:
        """Return the list of crop names from the training dataset."""
        return sorted(self._crop_names)
