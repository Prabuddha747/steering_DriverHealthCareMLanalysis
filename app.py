"""
Streamlit Driver Health Model Comparison App
Compares LSTM, LSTM-AE, and XGBoost for stress classification.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "synthetic_busroute_driver_sensors.csv"
WINDOW_SIZE = 30
HIDDEN_SIZE = 64

st.set_page_config(
    page_title="Driver Health Model Comparison",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    path = ARTIFACTS_DIR / "artifacts.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_lstmae_model():
    model_path = MODELS_DIR / "lstm_ae.pt"
    prep_path = MODELS_DIR / "preprocess.pkl"
    if not model_path.exists() or not prep_path.exists():
        return None, None
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)
    return ckpt, prep


def load_data(limit=None):
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, nrows=limit)
    return df


def render_sidebar():
    st.sidebar.title("Driver Health Analysis")
    st.sidebar.markdown("---")
    section = st.sidebar.radio(
        "Navigate",
        [
            "Home",
            "Training & Validation Loss",
            "ROC Curve",
            "Adaptive Thresholding",
            "Performance Metrics Table",
            "F1-Score per Class",
            "Confusion Matrices",
            "Other Analyses",
            "LSTM-AE Prediction",
        ],
    )
    return section


def section_home(artifacts, df):
    st.title("Driver Health Model Comparison")
    st.markdown(
        "Compare **LSTM**, **LSTM-AE**, and **XGBoost** for driver stress classification. "
        "LSTM-AE is our main model."
    )
    st.markdown("---")

    if df is not None:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Features", len([c for c in df.columns if c not in ["SNo", "StressLabel", "Date", "Timestamp"]]))
        with col3:
            st.metric("Classes", df["StressLabel"].nunique() if "StressLabel" in df.columns else 0)
        st.markdown("**Classes:** " + ", ".join(df["StressLabel"].unique().tolist()) if "StressLabel" in df.columns else "")

    if artifacts:
        st.subheader("Precomputed Results")
        st.info("Artifacts loaded. Use the sidebar to explore training curves, ROC, metrics, and confusion matrices.")
    else:
        st.warning(
            "No artifacts found. Run `python scripts/train_and_save_artifacts.py` to generate training results."
        )


def section_loss_curves(artifacts):
    st.title("Training & Validation Loss Curves")
    if not artifacts or "loss_curves" not in artifacts:
        st.warning("No loss curves found. Run the training script first.")
        return

    loss = artifacts["loss_curves"]
    fig = go.Figure()
    for model_name, hist in loss.items():
        epochs = range(1, len(hist["loss"]) + 1)
        fig.add_trace(go.Scatter(x=list(epochs), y=hist["loss"], name=f"{model_name} Train", mode="lines"))
        fig.add_trace(go.Scatter(x=list(epochs), y=hist["val_loss"], name=f"{model_name} Val", mode="lines", line=dict(dash="dash")))
    fig.update_layout(title="Loss vs Epochs", xaxis_title="Epochs", yaxis_title="Loss", hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)


def section_roc(artifacts):
    st.title("ROC Curve: LSTM Variants Comparison")
    if not artifacts or "roc" not in artifacts:
        st.warning("No ROC data found. Run the training script first.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dot", color="gray")))
    for roc in artifacts["roc"]:
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name=f"{roc['model']} (AUC={roc['auc']:.3f})", mode="lines"))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def section_adaptive_thresholding(artifacts):
    st.title("Performance Improvement with Adaptive Thresholding")
    if not artifacts or "metrics" not in artifacts:
        st.warning("No metrics found. Run the training script first.")
        return

    m = artifacts["metrics"]
    models = list(m.keys())
    acc_no = [m[mod]["without_at"]["accuracy"] * 100 for mod in models]
    acc_at = [m[mod]["with_at"]["accuracy"] * 100 for mod in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=acc_no, name="Without AT", marker_color="lightgray"))
    fig.add_trace(go.Bar(x=models, y=acc_at, name="With AT", marker_color="coral"))
    fig.update_layout(barmode="group", yaxis_title="Accuracy (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)


def section_metrics_table(artifacts):
    st.title("Performance Metrics Across Models")
    if not artifacts or "metrics" not in artifacts:
        st.warning("No metrics found. Run the training script first.")
        return

    rows = []
    for model_name, data in artifacts["metrics"].items():
        for at_name, metrics in data.items():
            if at_name != "without_at":
               continue  # Only show rows without Adaptive Thresholding
            label = "With AT" if at_name == "with_at" else "Without AT"
            rows.append({
                "Model": model_name,
                # "Adaptive Thresholding": label,
                "Accuracy": f"{metrics['accuracy']*100:.1f}%",
                "Precision": f"{metrics['precision']*100:.1f}%",
                "Recall": f"{metrics['recall']*100:.1f}%",
                "F1-Score": f"{metrics['f1']:.3f}",
                "AUC-ROC": f"{metrics['auc_roc']:.3f}",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def section_f1_per_class(artifacts):
    st.title("Multi-class F1-Score per Class")
    if not artifacts or "f1_per_class" not in artifacts:
        st.warning("No F1 per class found. Run the training script first.")
        return

    f1_data = artifacts["f1_per_class"]
    class_names = artifacts.get("class_names", list(next(iter(f1_data.values())).keys()))
    models = list(f1_data.keys())

    fig = go.Figure()
    for model_name, scores in f1_data.items():
        fig.add_trace(go.Bar(name=model_name, x=class_names, y=[scores.get(c, 0) for c in class_names]))
    fig.update_layout(barmode="group", yaxis_title="F1-Score", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    rows = []
    for cls in class_names:
        row = {"Class": cls}
        for mod in models:
            row[mod] = f"{f1_data[mod].get(cls, 0):.3f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def section_confusion_matrices(artifacts):
    st.title("Confusion Matrices")
    if not artifacts or "confusion_matrices" not in artifacts:
        st.warning("No confusion matrices found. Run the training script first.")
        return

    cm_data = artifacts["confusion_matrices"]
    class_names = artifacts.get("class_names", [f"Class {i}" for i in range(4)])

    for model_name, cm in cm_data.items():
        with st.expander(f"**{model_name}**", expanded=(model_name == "LSTM-AE")):
            fig = px.imshow(
                cm,
                x=class_names,
                y=class_names,
                labels=dict(x="Predicted", y="True"),
                color_continuous_scale="Blues",
                text_auto=True,
            )
            fig.update_layout(title=f"{model_name} Confusion Matrix", height=400)
            st.plotly_chart(fig, use_container_width=True)


def section_other_analyses(artifacts, df):
    st.title("Other Analyses")
    if df is None or not DATA_PATH.exists():
        st.warning("Data not found for additional analyses.")
        return

    tab1, tab2, tab3 = st.tabs(["Feature Correlation Heatmap", "Feature Distribution", "Class Distribution"])

    with tab1:
        num_cols = ["HeartRate_bpm", "SpO2_pct", "BodyTemp_C", "GSR_uS", "Speed_kmph"]
        num_cols = [c for c in num_cols if c in df.columns]
        if num_cols:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(title="Feature Correlation Heatmap", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        feat = st.selectbox("Feature", ["HeartRate_bpm", "SpO2_pct", "BodyTemp_C", "GSR_uS", "Speed_kmph"], key="hist_feat")
        if feat in df.columns and "StressLabel" in df.columns:
            fig = px.histogram(df, x=feat, color="StressLabel", barmode="overlay", opacity=0.7)
            fig.update_layout(title=f"Distribution of {feat} by Stress Label", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "StressLabel" in df.columns:
            counts = df["StressLabel"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, title="Class Distribution")
            st.plotly_chart(fig, use_container_width=True)


def section_prediction(ckpt, prep, df):
    st.title("LSTM-AE Prediction (Main Model)")
    if ckpt is None or prep is None:
        st.warning("LSTM-AE model not found. Run `python scripts/train_and_save_artifacts.py` first.")
        return

    st.info("Use the LSTM-AE model to predict stress level from sensor data.")

    option = st.radio("Input", ["Use sample from dataset", "Upload CSV"])
    X_in = None
    sample_info = ""

    if option == "Use sample from dataset":
        if df is not None and len(df) >= WINDOW_SIZE:
            n = st.slider("Number of samples to predict", 1, 20, 5)
            # Build sequences from df
            df_sample = df.head(5000)
            le_activity = prep["le_activity"]
            le_route = prep["le_route"]
            scaler = prep["scaler"]
            num_cols = ["HeartRate_bpm", "SpO2_pct", "BodyTemp_C", "GSR_uS", "Speed_kmph", "Latitude", "Longitude", "DayIndex"]
            done = False
            for driver_id, g in df_sample.groupby("DriverID"):
                if done:
                    break
                g = g.sort_values("Timestamp").reset_index(drop=True)
                if len(g) < WINDOW_SIZE:
                    continue
                X_raw = np.hstack([
                    g[num_cols].values.astype(np.float32),
                    le_activity.transform(g["ActivityState"].astype(str)).reshape(-1, 1).astype(np.float32),
                    le_route.transform(g["RouteDirection"].astype(str)).reshape(-1, 1).astype(np.float32),
                ])
                feats = scaler.transform(X_raw)
                for i in range(len(g) - WINDOW_SIZE):
                    seq = feats[i : i + WINDOW_SIZE]
                    if X_in is None:
                        X_in = seq[np.newaxis, ...]
                    else:
                        X_in = np.vstack([X_in, seq[np.newaxis, ...]])
                    if X_in.shape[0] >= n:
                        X_in = X_in[:n]
                        done = True
                        break
        else:
            st.warning("Dataset not loaded or too small.")
    else:
        uploaded = st.file_uploader("Upload CSV with same columns as training data", type=["csv"])
        if uploaded:
            up_df = pd.read_csv(uploaded)
            if len(up_df) < WINDOW_SIZE:
                st.error(f"Need at least {WINDOW_SIZE} rows.")
            else:
                le_activity = prep["le_activity"]
                le_route = prep["le_route"]
                scaler = prep["scaler"]
                num_cols = ["HeartRate_bpm", "SpO2_pct", "BodyTemp_C", "GSR_uS", "Speed_kmph", "Latitude", "Longitude", "DayIndex"]
                X_raw = np.hstack([
                    up_df[num_cols].values.astype(np.float32),
                    le_activity.transform(up_df["ActivityState"].astype(str)).reshape(-1, 1).astype(np.float32),
                    le_route.transform(up_df["RouteDirection"].astype(str)).reshape(-1, 1).astype(np.float32),
                ])
                feats = scaler.transform(X_raw)
                seq = feats[-WINDOW_SIZE:][np.newaxis, ...]
                X_in = seq
                sample_info = f"Uploaded file: {uploaded.name}"

    if X_in is not None:
        # Load model and predict
        class LSTMAEModel(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                self.encoder = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
                self.decoder = torch.nn.LSTM(hidden_size, input_size, batch_first=True)
                self.fc = torch.nn.Linear(hidden_size, num_classes)
                self.hidden_size = hidden_size

            def forward(self, x):
                _, (h, _) = self.encoder(x)
                return self.fc(h[-1])

        model = LSTMAEModel(ckpt["input_size"], ckpt["hidden_size"], ckpt["num_classes"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            X_t = torch.FloatTensor(X_in)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)

        le_label = prep["le_label"]
        class_names = list(le_label.classes_)

        st.subheader("Predictions")
        for i, (p, proba) in enumerate(zip(preds, probs)):
            st.write(f"**Sample {i+1}:** {class_names[p]} (confidence: {proba[p]:.2%})")
        st.dataframe(pd.DataFrame({"Predicted": [class_names[p] for p in preds], "Confidence": [probs[i, p] for i, p in enumerate(preds)]}), use_container_width=True, hide_index=True)


def main():
    section = render_sidebar()
    artifacts = load_artifacts()
    ckpt, prep = load_lstmae_model()
    df = load_data(limit=144000)

    if section == "Home":
        section_home(artifacts, df)
    elif section == "Training & Validation Loss":
        section_loss_curves(artifacts)
    elif section == "ROC Curve":
        section_roc(artifacts)
    elif section == "Adaptive Thresholding":
        section_adaptive_thresholding(artifacts)
    elif section == "Performance Metrics Table":
        section_metrics_table(artifacts)
    elif section == "F1-Score per Class":
        section_f1_per_class(artifacts)
    elif section == "Confusion Matrices":
        section_confusion_matrices(artifacts)
    elif section == "Other Analyses":
        section_other_analyses(artifacts, df)
    elif section == "LSTM-AE Prediction":
        section_prediction(ckpt, prep, df)


if __name__ == "__main__":
    main()
