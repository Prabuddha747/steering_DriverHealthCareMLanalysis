"""
Shared analysis utilities: metrics, ROC, confusion matrix, F1 per class.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute accuracy, precision, recall, F1, AUC."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc = (
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        if y_proba is not None
        else 0.0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


def roc_curve_data(y_true, y_proba, model_name):
    """Build ROC curve data for plotting."""
    n_classes = y_proba.shape[1]
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc), "model": model_name}


def f1_per_class(y_true, y_pred, class_names):
    """F1 score per class."""
    r = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {k: float(r[k].get("f1-score", 0)) for k in class_names if k in r}


def get_confusion_matrix(y_true, y_pred):
    """Return confusion matrix as list."""
    return confusion_matrix(y_true, y_pred).tolist()
