"""
Crop Recommendation — Model Training Script
=============================================

Trains and evaluates two classification models (Random Forest and XGBoost)
on the crop recommendation dataset, then saves the best performer to models/.

Run with:
    python training/train_crop_recommendation.py

Dataset: data/crop_recommendation_data.csv
    • Features: N, P, K, temperature, humidity, ph, rainfall
    • Target:   label (crop name — 22 classes, balanced at 100 each)
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
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "crop_recommendation_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommendation_model.joblib")

# ── Constants ────────────────────────────────────────────────────────

TARGET_COL = "label"

# All 7 features are numeric — no categorical encoding needed
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> pd.DataFrame:
    """Load and validate the crop recommendation dataset."""
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Validate required columns
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Check for null values
    null_count = df[required].isnull().sum().sum()
    print(f"Null values: {null_count}")
    if null_count > 0:
        print("WARNING: Dropping rows with null values")
        df = df.dropna(subset=required)
        print(f"After dropping: {df.shape[0]} rows")

    # Show target distribution (class balance)
    print(f"\nTarget classes ({df[TARGET_COL].nunique()}):")
    print(df[TARGET_COL].value_counts().to_string())

    return df


def train_and_evaluate(name: str, model, X_train, X_test, y_train, y_test,
                        target_encoder: LabelEncoder) -> dict:
    """Train a model, evaluate it, and print metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Decode predictions for the classification report (human-readable crop names)
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
    print("  CropGenie — Crop Recommendation Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    X = df[FEATURE_COLS].copy()
    y_raw = df[TARGET_COL].copy()

    # 2. Encode crop labels to integers (LabelEncoder stores the mapping)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"\nEncoded {len(label_encoder.classes_)} crop classes: "
          f"{label_encoder.classes_.tolist()}")

    # 3. Train/test split (stratified to keep class balance in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. Train both models (XGBoost is optional)
    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("\nNOTE: XGBoost not installed — training Random Forest only.")

    results = []

    # Random Forest — primary model
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    results.append(
        train_and_evaluate("Random Forest", rf, X_train, X_test,
                           y_train, y_test, label_encoder)
    )

    # XGBoost (if available)
    if has_xgboost:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        results.append(
            train_and_evaluate("XGBoost", xgb, X_train, X_test,
                               y_train, y_test, label_encoder)
        )

    # 5. Pick the best model by accuracy
    best_result = max(results, key=lambda r: r["accuracy"])
    best_name = best_result["name"]

    print(f"\n{'=' * 60}")
    print(f"  Best model: {best_name} (Accuracy = {best_result['accuracy']:.4f})")
    print(f"{'=' * 60}")

    # 6. Re-train the best model on full data for production use
    if best_name == "XGBoost":
        best_model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    else:
        best_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    best_model.fit(X, y)

    # 7. Save model + encoder + metadata (same artifact-dict pattern)
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifact = {
        "model": best_model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "label_encoder": label_encoder,
        "metrics": best_result,
        "model_name": best_name,
        # Store the list of supported crops for validation at prediction time
        "crop_names": sorted(label_encoder.classes_.tolist()),
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"  Model:    {best_name}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Target:   {TARGET_COL}")
    print(f"  Crops:    {artifact['crop_names']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
