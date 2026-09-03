"""
Fertilizer recommendation service — loads the trained ML classifier and
generates fertilizer predictions, then optionally delegates to Gemini
for a farmer-friendly explanation.

The model file (fertilizer_model.joblib) is produced by
``training/train_fertilizer.py`` and loaded once at service startup.
"""

import logging
import os
from typing import Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# ── Model path ───────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fertilizer_model.joblib")


class FertilizerService:
    """Loads the trained fertilizer classifier and runs predictions."""

    def __init__(self) -> None:
        """Load the saved model artifact from disk."""
        if not os.path.exists(_MODEL_PATH):
            raise RuntimeError(
                "Trained fertilizer model not found. Run "
                "`python training/train_fertilizer.py` first to generate "
                f"the model at {_MODEL_PATH}."
            )

        # NOTE (security): joblib.load deserialises Python objects.
        # The model file is produced by our own training script and stored
        # locally — it is trusted provenance.  Do NOT load model files from
        # untrusted external sources.
        artifact = joblib.load(_MODEL_PATH)
        self._model = artifact["model"]
        self._feature_cols: list[str] = artifact["feature_cols"]
        self._numeric_features: list[str] = artifact["numeric_features"]
        self._categorical_features: list[str] = artifact["categorical_features"]
        self._encoders: dict = artifact["encoders"]
        self._model_name: str = artifact.get("model_name", "unknown")
        self._accuracy: float = artifact.get("metrics", {}).get("accuracy", 0.0)

        # Training-set metadata for validation
        self._soil_types: list[str] = artifact.get("soil_types", [])
        self._crop_types: list[str] = artifact.get("crop_types", [])
        self._fertilizer_names: list[str] = artifact.get("fertilizer_names", [])

        logger.info(
            "FertilizerService loaded: %s (accuracy=%.4f, features=%s)",
            self._model_name,
            self._accuracy,
            self._feature_cols,
        )

    # ── Public API ───────────────────────────────────────────────────

    def predict(
        self,
        temperature: float,
        humidity: float,
        moisture: float,
        soil_type: str,
        crop_type: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
    ) -> str:
        """
        Predict the recommended fertilizer name.

        Parameters
        ----------
        temperature, humidity, moisture : float
            Environmental conditions.
        soil_type, crop_type : str
            Must match one of the training-set categories.
        nitrogen, phosphorus, potassium : float
            Current NPK soil levels.

        Returns
        -------
        str
            Predicted fertilizer name (e.g. 'Urea', 'DAP', '14-35-14').

        Raises
        ------
        ValueError
            If a categorical input is not in the training-set vocabulary.
        """
        # Validate categorical inputs
        if soil_type not in self._soil_types:
            raise ValueError(
                f"Unknown soil type '{soil_type}'. "
                f"Supported: {', '.join(self._soil_types)}"
            )
        if crop_type not in self._crop_types:
            raise ValueError(
                f"Unknown crop type '{crop_type}'. "
                f"Supported: {', '.join(self._crop_types)}"
            )

        # Encode categorical features
        soil_enc = self._encoders["Soil Type"].transform([soil_type])[0]
        crop_enc = self._encoders["Crop Type"].transform([crop_type])[0]

        # Build feature row — must match feature_cols order from training
        row = {
            "Temperature": temperature,
            "Humidity": humidity,
            "Moisture": moisture,
            "Nitrogen": nitrogen,
            "Potassium": potassium,
            "Phosphorus": phosphorus,
            "Soil Type_enc": soil_enc,
            "Crop Type_enc": crop_enc,
        }
        features = pd.DataFrame([row], columns=self._feature_cols)

        # Predict encoded class
        pred_encoded = self._model.predict(features)[0]

        # Decode back to fertilizer name
        fertilizer_name = self._encoders["Fertilizer Name"].inverse_transform(
            [pred_encoded]
        )[0]
        return fertilizer_name

    def get_supported_crops(self) -> list[str]:
        """Return the list of crop types from the training dataset."""
        return sorted(self._crop_types)

    def get_supported_soils(self) -> list[str]:
        """Return the list of soil types from the training dataset."""
        return sorted(self._soil_types)

    def get_supported_fertilizers(self) -> list[str]:
        """Return the list of fertilizer names from the training dataset."""
        return sorted(self._fertilizer_names)
