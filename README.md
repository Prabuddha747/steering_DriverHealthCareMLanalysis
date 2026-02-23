# Driver Health Model Comparison

Streamlit app comparing **LSTM**, **LSTM-AE**, and **XGBoost** for driver stress classification on bus route sensor data.

## Setup

```bash
# Create virtual environment (use --without-pip if you hit UnicodeDecodeError on external drives)
python3 -m venv venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./venv/bin/python get-pip.py
rm get-pip.py

# Install dependencies
./venv/bin/pip install -r requirements.txt

# Activate venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### If you see `UnicodeDecodeError` with pip or Python

This can occur when the project lives on an external drive (e.g. PV_HDD). Options:

1. **Use `--without-pip` + get-pip.py** (see above) — avoids the error when installing pip.
2. **Move the project to your main drive** (e.g. `~/Projects/`) and create the venv there.
3. **Use system Python with `--user`**:  
   `pip3 install --user -r requirements.txt` (no venv).

## Data

Place `synthetic_busroute_driver_sensors.csv` in the project root. The CSV must include:

- `HeartRate_bpm`, `SpO2_pct`, `BodyTemp_C`, `GSR_uS`, `Speed_kmph`, `Latitude`, `Longitude`, `DayIndex`
- `ActivityState`, `RouteDirection`
- `StressLabel` (target: Low, Med, High)

## Generate Artifacts

**Mock artifacts** are included so you can run the app immediately for demo. For full functionality (LSTM-AE prediction), run the training script:

```bash
python scripts/train_and_save_artifacts.py
```

This creates:

- `artifacts/artifacts.json` — loss curves, ROC, metrics, confusion matrices (overwrites mock)
- `models/lstm_ae.pt` — LSTM-AE model weights
- `models/preprocess.pkl` — scaler and encoders for prediction

## Run the App

```bash
streamlit run app.py
```

## Sections

- **Home** — Dataset overview
- **Training & Validation Loss** — Loss curves for LSTM and LSTM-AE
- **ROC Curve** — ROC comparison with AUC
- **Adaptive Thresholding** — Accuracy with/without adaptive thresholding
- **Performance Metrics Table** — Accuracy, precision, recall, F1, AUC-ROC
- **F1-Score per Class** — Per-class F1 for each model
- **Confusion Matrices** — Confusion matrices for all models
- **Other Analyses** — Feature correlation heatmap, histograms, class distribution
- **LSTM-AE Prediction** — Run LSTM-AE inference (sample or upload CSV)

## Git Repository

Project is set up for [steeringMLanalysis](https://github.com/Prabuddha747/steeringMLanalysis.git).
