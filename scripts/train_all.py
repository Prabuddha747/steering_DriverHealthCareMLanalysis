"""
Orchestrator: train all three models and aggregate artifacts for the Streamlit app.
Run: python scripts/train_all.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.config import ARTIFACTS_DIR, MODELS_DIR
from scripts.train_lstm import run as run_lstm
from scripts.train_lstm_ae import run as run_lstm_ae
from scripts.train_xgboost import run as run_xgboost


def to_serializable(obj):
    """Convert numpy types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training LSTM...")
    print("=" * 60)
    lstm_results = run_lstm()

    print("\n" + "=" * 60)
    print("Training LSTM-AE...")
    print("=" * 60)
    lstmae_results = run_lstm_ae()

    print("\n" + "=" * 60)
    print("Training XGBoost...")
    print("=" * 60)
    xgb_results = run_xgboost()

    # Aggregate artifacts for Streamlit app
    artifacts = {
        "loss_curves": {
            "LSTM": lstm_results["hist"],
            "LSTM-AE": lstmae_results["hist"],
        },
        "roc": [
            lstm_results["roc"],
            lstmae_results["roc"],
            xgb_results["roc"],
        ],
        "metrics": {
            "LSTM": {"without_at": lstm_results["metrics"], "with_at": lstm_results["metrics"]},
            "LSTM-AE": {"without_at": lstmae_results["metrics"], "with_at": lstmae_results["metrics"]},
            "XGBoost": {"without_at": xgb_results["metrics"], "with_at": xgb_results["metrics"]},
        },
        "f1_per_class": {
            "LSTM": lstm_results["f1_per_class"],
            "LSTM-AE": lstmae_results["f1_per_class"],
            "XGBoost": xgb_results["f1_per_class"],
        },
        "confusion_matrices": {
            "LSTM": lstm_results["confusion_matrix"],
            "LSTM-AE": lstmae_results["confusion_matrix"],
            "XGBoost": xgb_results["confusion_matrix"],
        },
        "class_names": list(lstm_results["f1_per_class"].keys()),
    }

    with open(ARTIFACTS_DIR / "artifacts.json", "w") as f:
        json.dump(to_serializable(artifacts), f, indent=2)

    print("\n" + "=" * 60)
    print(f"Artifacts saved to {ARTIFACTS_DIR}")
    print(f"Models saved to {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
