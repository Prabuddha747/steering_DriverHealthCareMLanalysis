"""
Print exactly how the dataset is split into train vs validation.
Run: python scripts/inspect_split.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.config import (
    DATA_PATH,
    SAMPLE_SIZE,
    RANDOM_STATE,
    USE_TRIP_LEVEL,
    SPLIT_BY_DRIVER,
    WINDOW_SIZE,
    TRIP_SEQ_LENGTH,
)
from scripts.data import (
    _build_trip_ids,
    _aggregate_trips,
    _split_by_driver,
    load_and_preprocess,
)


def main():
    print("=" * 60)
    print("HOW THE DATASET IS SPLIT FOR TRAINING")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  USE_TRIP_LEVEL   = {USE_TRIP_LEVEL}  (one row per one-way FWD/BWD trip)")
    print(f"  SPLIT_BY_DRIVER  = {SPLIT_BY_DRIVER}  (train/val = different drivers)")
    print(f"  SAMPLE_SIZE      = {SAMPLE_SIZE}  (None = use all rows)")
    print(f"  RANDOM_STATE     = {RANDOM_STATE}")

    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    print(f"\n1. Raw CSV: {len(df):,} rows, {df['DriverID'].nunique()} drivers")
    print(f"   Drivers: {sorted(df['DriverID'].unique().tolist())}")

    if SAMPLE_SIZE:
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"\n2. After random sample of {SAMPLE_SIZE:,} rows: {len(df):,} rows")
        print(f"   (Same rows every run because RANDOM_STATE is fixed)")

    if USE_TRIP_LEVEL:
        df = _build_trip_ids(df)
        df = _aggregate_trips(df)
        df = df.sort_values(["DriverID", "Date", "trip_uid"]).reset_index(drop=True)
        print(f"\n3. After trip aggregation: {len(df):,} rows (one per one-way trip)")
    else:
        df = df.sort_values(["DriverID", "Timestamp"]).reset_index(drop=True)
        print(f"\n3. Row-level (no trip aggregation): {len(df):,} rows")

    if SPLIT_BY_DRIVER:
        train_mask, val_mask, train_drivers, val_drivers = _split_by_driver(df, "DriverID", 0.2)
        train_drivers = sorted(train_drivers)
        val_drivers = sorted(val_drivers)
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        print(f"\n4. Split BY DRIVER (no driver appears in both train and val):")
        print(f"   Train drivers ({len(train_drivers)}): {train_drivers}")
        print(f"   Val drivers   ({len(val_drivers)}):   {val_drivers}")
        print(f"   Train samples: {n_train:,}")
        print(f"   Val samples:   {n_val:,}")
    else:
        print(f"\n4. Split: random 80/20 (same driver can be in both train and val)")

    # Load via actual pipeline to confirm sequence counts
    data = load_and_preprocess()
    print(f"\n5. What the training scripts actually receive:")
    print(f"   X_seq_tr:  {data['X_seq_tr'].shape}")
    print(f"   X_seq_val: {data['X_seq_val'].shape}")
    print(f"   X_tab_tr:  {data['X_tab_tr'].shape}")
    print(f"   X_tab_val: {data['X_tab_val'].shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
