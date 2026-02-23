"""
Train XGBoost model on driver sensor tabular data.
Saves model to models/, runs analysis, returns results for aggregation.
"""
import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import xgboost as xgb

from scripts.config import MODELS_DIR, ARTIFACTS_DIR, RANDOM_STATE
from scripts.data import load_and_preprocess
from scripts.analysis import (
    compute_metrics,
    roc_curve_data,
    f1_per_class,
    get_confusion_matrix,
)


def run():
    """Load data, train XGBoost, run analysis, save model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing...")
    data = load_and_preprocess()
    X_tr = data["X_tab_tr"]
    X_val = data["X_tab_val"]
    y_tr = data["y_tab_tr"]
    y_val = data["y_tab_val"]
    class_names = data["class_names"]

    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        reg_alpha=0.15,
        reg_lambda=1.5,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_val)
    pred = np.argmax(proba, axis=1)

    metrics = compute_metrics(y_val, pred, proba)
    roc = roc_curve_data(y_val, proba, "XGBoost")
    f1_per_cls = f1_per_class(y_val, pred, class_names)
    cm = get_confusion_matrix(y_val, pred)

    # Save model
    with open(MODELS_DIR / "xgboost.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"XGBoost model saved to {MODELS_DIR / 'xgboost.pkl'}")

    return {
        "hist": None,
        "metrics": metrics,
        "roc": roc,
        "f1_per_class": f1_per_cls,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    run()
