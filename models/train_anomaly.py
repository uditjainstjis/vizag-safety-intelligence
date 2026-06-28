"""
Train IsolationForest on normal operations data.

Usage:
    python models/train_anomaly.py

If /data/normal_ops.csv does not exist, synthetic normal-operations data
is generated automatically.  The artifact (model + scaler + feature list)
is saved to models/iso_forest.pkl.
"""

import os
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ #
# Paths                                                               #
# ------------------------------------------------------------------ #
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "normal_ops.csv"
MODEL_PATH = BASE_DIR / "models" / "iso_forest.pkl"

FEATURES = ["h2s_ppm", "co_ppm", "pressure_kpa", "temperature_c", "vibration_ms2"]


# ------------------------------------------------------------------ #
# Data loading / generation                                           #
# ------------------------------------------------------------------ #
def load_or_generate_data() -> pd.DataFrame:
    """
    Load real normal-operations data if available, otherwise generate a
    synthetic dataset representative of safe steady-state plant readings.
    """
    if DATA_PATH.exists():
        print(f"Loading real data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required feature columns: {missing}"
            )
        return df[FEATURES].dropna()

    print("No real data found — generating synthetic normal-operations dataset…")
    np.random.seed(42)
    n = 500

    # Normal operating ranges based on Vizag process spec
    data = {
        "h2s_ppm":       np.random.normal(10.0, 2.0, n).clip(2,   20),
        "co_ppm":        np.random.normal(25.0, 5.0, n).clip(5,   50),
        "pressure_kpa":  np.random.normal(785,  8.0, n).clip(760, 810),
        "temperature_c": np.random.normal(390,  5.0, n).clip(375, 405),
        "vibration_ms2": np.random.normal(1.2,  0.3, n).clip(0.3, 2.5),
    }
    df = pd.DataFrame(data)

    # Save synthetic data so future runs are reproducible
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset ({n} rows) saved to {DATA_PATH}")
    return df


# ------------------------------------------------------------------ #
# Training                                                            #
# ------------------------------------------------------------------ #
def train() -> dict:
    """
    Train an IsolationForest on normal-operations data and persist the
    artifact {model, scaler, features} to models/iso_forest.pkl.

    Returns the artifact dict.
    """
    df = load_or_generate_data()
    X  = df[FEATURES].values

    print(f"Training on {len(X)} samples with features: {FEATURES}")

    # Standardise features before fitting
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # IsolationForest — 5% contamination means we expect ≤5% anomalies in training
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Quick sanity check — score the training set itself
    scores = model.score_samples(X_scaled)
    print(
        f"Training score_samples — min: {scores.min():.4f}  "
        f"mean: {scores.mean():.4f}  max: {scores.max():.4f}"
    )

    # Identify fraction labelled as anomaly on training data
    preds    = model.predict(X_scaled)
    anomaly_frac = (preds == -1).mean()
    print(f"Anomaly fraction on training data: {anomaly_frac:.2%} "
          f"(target ≤ contamination={0.05:.0%})")

    # Persist artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "scaler": scaler, "features": FEATURES}
    joblib.dump(artifact, MODEL_PATH)
    print(f"Artifact saved to {MODEL_PATH}")

    return artifact


# ------------------------------------------------------------------ #
# Quick smoke-test                                                    #
# ------------------------------------------------------------------ #
def smoke_test(artifact: dict) -> None:
    """
    Verify the saved artifact can be reloaded and produces reasonable
    scores for a known-normal and a known-anomalous reading.
    """
    loaded   = joblib.load(MODEL_PATH)
    model    = loaded["model"]
    scaler   = loaded["scaler"]

    normal_reading  = [[10,  25, 785, 390, 1.2]]   # well within normal range
    anomaly_reading = [[48, 190, 848, 419, 5.8]]   # near / at alarm thresholds

    for label, sample in [("NORMAL", normal_reading), ("ANOMALY", anomaly_reading)]:
        scaled = scaler.transform(sample)
        raw    = model.score_samples(scaled)[0]
        pred   = model.predict(scaled)[0]
        flag   = "anomaly" if pred == -1 else "inlier"
        print(f"  Smoke-test {label:7s}: score_samples={raw:+.4f}  prediction={flag}")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    artifact = train()
    print("\nRunning smoke-test on saved model…")
    smoke_test(artifact)
    print("\nDone. IsolationForest artifact ready at:", MODEL_PATH)
