"""
Fertilizer Recommendation — Model Training Script
==================================================

Trains and evaluates two classification models (Random Forest and XGBoost)
on the fertilizer dataset, then saves the best performer to models/.

Run with:
    python training/train_fertilizer.py

Dataset: data/fertilizer_data.csv
    • Features:   Temperature, Humidity, Moisture, Soil Type, Crop Type,
                  Nitrogen, Potassium, Phosphorous
    • Target:     Fertilizer Name (7 classes)
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "fertilizer_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fertilizer_model.joblib")

# ── Constants ────────────────────────────────────────────────────────

TARGET_COL = "Fertilizer Name"

# Numeric feature columns (exact names after renaming typos)
NUMERIC_FEATURES = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorus"]

# Categorical feature columns
CATEGORICAL_FEATURES = ["Soil Type", "Crop Type"]

# Column renames — the CSV has typos: "Temparature", "Humidity " (trailing space)
RENAME_MAP = {"Temparature": "Temperature", "Humidity ": "Humidity", "Phosphorous": "Phosphorus"}

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> pd.DataFrame:
    """Load and validate the fertilizer dataset, fixing column typos."""
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Original columns: {df.columns.tolist()}")

    # Rename columns with typos
    df = df.rename(columns=RENAME_MAP)
    print(f"Renamed columns: {df.columns.tolist()}")

    # Validate required columns
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    print(f"Null values: {df[required].isnull().sum().sum()}")

    # Show target distribution
    print(f"\nTarget classes ({df[TARGET_COL].nunique()}):")
    print(df[TARGET_COL].value_counts().to_string())

    return df


def encode_features(df: pd.DataFrame) -> tuple:
    """
    Encode categorical features and target with LabelEncoder.

    Returns
    -------
    X : DataFrame — encoded feature matrix
    y : Series — encoded target
    encoders : dict — {column_name: LabelEncoder} for all encoded columns
    """
    encoders = {}

    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    df["target_enc"] = le_target.fit_transform(df[TARGET_COL])
    encoders[TARGET_COL] = le_target

    # Build feature matrix: numeric + encoded categorical
    feature_cols = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]
    X = df[feature_cols].copy()
    y = df["target_enc"].copy()

    return X, y, encoders, feature_cols


def train_and_evaluate(name: str, model, X_train, X_test, y_train, y_test,
                        target_encoder: LabelEncoder) -> dict:
    """Train a model, evaluate it, and print metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Decode predictions for the classification report
    y_test_decoded = target_encoder.inverse_transform(y_test)
    y_pred_decoded = target_encoder.inverse_transform(y_pred)

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

    metrics = {"name": name, "accuracy": accuracy}
    return metrics


def main():
    """Train models, pick the best, and save to disk."""
    print("\n" + "=" * 60)
    print("  CropGenie — Fertilizer Recommendation Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Encode features and target
    X, y, encoders, feature_cols = encode_features(df)
    print(f"\nFeature columns: {feature_cols}")
    print(f"Encoded samples:\n{X.head()}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. Train both models
    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("\nNOTE: XGBoost not installed — training Random Forest only.")

    target_encoder = encoders[TARGET_COL]
    results = []

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    results.append(
        train_and_evaluate("Random Forest", rf, X_train, X_test, y_train, y_test, target_encoder)
    )

    # XGBoost (if available)
    if has_xgboost:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        results.append(
            train_and_evaluate("XGBoost", xgb, X_train, X_test, y_train, y_test, target_encoder)
        )

    # 5. Pick the best model by accuracy
    best_result = max(results, key=lambda r: r["accuracy"])
    best_name = best_result["name"]

    print(f"\n{'=' * 60}")
    print(f"  Best model: {best_name} (Accuracy = {best_result['accuracy']:.4f})")
    print(f"{'=' * 60}")

    # Re-train the best model on full data for production use
    if best_name == "XGBoost":
        best_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    else:
        best_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    best_model.fit(X, y)

    # 6. Save model + encoders + metadata
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifact = {
        "model": best_model,
        "feature_cols": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoders": encoders,          # {col_name: LabelEncoder}
        "metrics": best_result,
        "model_name": best_name,
        # Store the unique values from the training set for validation
        "soil_types": sorted(df["Soil Type"].unique().tolist()),
        "crop_types": sorted(df["Crop Type"].unique().tolist()),
        "fertilizer_names": sorted(df[TARGET_COL].unique().tolist()),
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"  Model:       {best_name}")
    print(f"  Accuracy:    {best_result['accuracy']:.4f}")
    print(f"  Features:    {feature_cols}")
    print(f"  Target:      {TARGET_COL}")
    print(f"  Soil types:  {artifact['soil_types']}")
    print(f"  Crop types:  {artifact['crop_types']}")
    print(f"  Fertilizers: {artifact['fertilizer_names']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
