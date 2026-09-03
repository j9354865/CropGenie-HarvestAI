"""
Crop Yield Prediction — Model Training Script
==============================================

Trains and evaluates two regression models (Random Forest and XGBoost)
on the crop yield dataset, then saves the best performer to models/.

Run with:
    python training/train_model.py

Dataset: data/crop_yield_data.csv
    • Features: Fertilizer, temp, N (Nitrogen), P (Phosphorus), K (Potassium)
    • Target:   yeild (crop yield per unit area)
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "crop_yield_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_yield_model.joblib")

# ── Constants ────────────────────────────────────────────────────────

TARGET_COL = "yeild"            # column name in the CSV (note spelling)
FEATURE_COLS = ["Fertilizer", "temp", "N", "P", "K"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> pd.DataFrame:
    """Load and validate the crop yield dataset."""
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

    print(f"Null values: {df[required].isnull().sum().sum()}")
    return df


def train_and_evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train a model, evaluate it, and print metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"name": name, "r2": r2, "mae": mae, "rmse": rmse}

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  RMSE     : {rmse:.4f}")

    return metrics


def main():
    """Train models, pick the best, and save to disk."""
    print("\n" + "=" * 60)
    print("  CropGenie — Crop Yield Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Train both models
    try:
        from xgboost import XGBRegressor
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("\nNOTE: XGBoost not installed — training Random Forest only.")

    results = []

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    results.append(
        train_and_evaluate("Random Forest", rf, X_train, X_test, y_train, y_test)
    )

    # XGBoost (if available)
    if has_xgboost:
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        results.append(
            train_and_evaluate("XGBoost", xgb, X_train, X_test, y_train, y_test)
        )

    # 4. Pick the best model by R²
    best_result = max(results, key=lambda r: r["r2"])
    best_name = best_result["name"]

    print(f"\n{'=' * 60}")
    print(f"  Best model: {best_name} (R² = {best_result['r2']:.4f})")
    print(f"{'=' * 60}")

    # Re-train the best model on full data for production use
    if best_name == "XGBoost":
        best_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        best_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    best_model.fit(X, y)

    # 5. Save model + metadata
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifact = {
        "model": best_model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "metrics": best_result,
        "model_name": best_name,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"  Model:    {best_name}")
    print(f"  R²:       {best_result['r2']:.4f}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Target:   {TARGET_COL}")
    print("\nDone.")


if __name__ == "__main__":
    main()
