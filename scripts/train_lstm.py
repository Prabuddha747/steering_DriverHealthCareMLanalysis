"""
Train LSTM model on driver sensor data.
Saves model to models/, runs analysis, and returns results for aggregation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from scripts.config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    EPOCHS,
    BATCH_SIZE,
    HIDDEN_SIZE,
    DROPOUT,
    WEIGHT_DECAY,
)
from scripts.data import load_and_preprocess
from scripts.models.lstm_model import LSTMModel
from scripts.analysis import (
    compute_metrics,
    roc_curve_data,
    f1_per_class,
    get_confusion_matrix,
)


def train_lstm(X_train, y_train, X_val, y_val, num_classes, device):
    """Train LSTM model and return model + history."""
    model = LSTMModel(X_train.shape[2], HIDDEN_SIZE, num_classes, dropout=DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.LongTensor(y_val).to(device)

    hist = {"loss": [], "val_loss": []}
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(Xt))
        total_loss = 0
        count = 0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = criterion(out, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            count += 1
        hist["loss"].append(total_loss / max(count, 1))

        model.eval()
        with torch.no_grad():
            vout = model(Xv)
            vloss = criterion(vout, yv).item()
        hist["val_loss"].append(vloss)

    return model, hist


def run():
    """Load data, train LSTM, run analysis, save model and metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing...")
    data = load_and_preprocess()
    X_tr = data["X_seq_tr"]
    X_val = data["X_seq_val"]
    y_tr = data["y_seq_tr"]
    y_val = data["y_seq_val"]
    num_classes = data["num_classes"]
    class_names = data["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Training LSTM...")

    model, hist = train_lstm(X_tr, y_tr, X_val, y_val, num_classes, device)
    model.eval()

    with torch.no_grad():
        Xv = torch.FloatTensor(X_val).to(device)
        proba = torch.softmax(model(Xv), dim=1).cpu().numpy()
    pred = np.argmax(proba, axis=1)

    metrics = compute_metrics(y_val, pred, proba)
    roc = roc_curve_data(y_val, proba, "LSTM")
    f1_per_cls = f1_per_class(y_val, pred, class_names)
    cm = get_confusion_matrix(y_val, pred)

    # Save model
    ckpt = {
        "model_state": model.state_dict(),
        "input_size": X_tr.shape[2],
        "hidden_size": HIDDEN_SIZE,
        "num_classes": num_classes,
    }
    torch.save(ckpt, MODELS_DIR / "lstm.pt")
    print(f"LSTM model saved to {MODELS_DIR / 'lstm.pt'}")

    return {
        "hist": hist,
        "metrics": metrics,
        "roc": roc,
        "f1_per_class": f1_per_cls,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    run()
