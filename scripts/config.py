"""
Shared configuration for driver health model training.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "synthetic_busroute_driver_sensors.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"

WINDOW_SIZE = 30
# One row per one-way trip (FWD or BWD) to reduce autocorrelation and overfitting
USE_TRIP_LEVEL = True
# Split by driver so validation is on unseen drivers (avoids leakage)
SPLIT_BY_DRIVER = True
SAMPLE_SIZE = 50_000  # Use subset for faster training; set None for full data
RANDOM_STATE = 42
EPOCHS = 25
BATCH_SIZE = 64
HIDDEN_SIZE = 64
DROPOUT = 0.35  # Reduces overfitting for more realistic validation metrics
WEIGHT_DECAY = 1e-2  # L2 regularization
# For trip-level sequences: how many consecutive trips per sample (per driver)
TRIP_SEQ_LENGTH = 10

FEATURE_COLS_NUM = [
    "HeartRate_bpm",
    "SpO2_pct",
    "BodyTemp_C",
    "GSR_uS",
    "Speed_kmph",
    "Latitude",
    "Longitude",
    "DayIndex",
]
FEATURE_COLS_CAT = ["ActivityState_enc", "RouteDirection_enc"]
