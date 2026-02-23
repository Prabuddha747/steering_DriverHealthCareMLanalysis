"""
EDA on driver sensor data with trip-level view.
FWD = source → destination, BWD = destination → source.
One trip = one contiguous one-way segment (same driver, date, direction).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.config import DATA_PATH
from scripts.data import _build_trip_ids, _aggregate_trips
from scripts.config import FEATURE_COLS_NUM


def main():
    print("Loading raw data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df):,}, Drivers: {df['DriverID'].nunique()}")
    print(f"  RouteDirection: {df['RouteDirection'].value_counts().to_dict()}")
    print(f"  StressLabel: {df['StressLabel'].value_counts().to_dict()}")

    print("\nBuilding trip segments (one trip = one-way FWD or BWD)...")
    df = _build_trip_ids(df)
    trip_counts = df.groupby("trip_uid").size()
    print(f"  Total trips: {len(trip_counts):,}")
    print(f"  Rows per trip: min={trip_counts.min()}, max={trip_counts.max()}, mean={trip_counts.mean():.1f}")

    print("\nAggregating to one row per trip (mean of sensors, mode of stress)...")
    trip_df = _aggregate_trips(df)
    print(f"  Trip-level rows: {len(trip_df):,}")

    # Trips per driver
    trips_per_driver = trip_df.groupby("DriverID").size()
    print(f"\nTrips per driver: min={trips_per_driver.min()}, max={trips_per_driver.max()}, mean={trips_per_driver.mean():.1f}")

    # Stress by direction
    print("\nStress distribution by direction (trip-level):")
    for direction in ["FWD", "BWD"]:
        sub = trip_df[trip_df["RouteDirection"] == direction]
        if len(sub) > 0:
            print(f"  {direction}: {sub['StressLabel'].value_counts().to_dict()}")

    # Numeric summary per trip
    print("\nAggregated sensor stats (mean per trip) - sample:")
    print(trip_df[FEATURE_COLS_NUM + ["RouteDirection", "StressLabel"]].describe().round(2).to_string())

    print("\nDone. Use trip-level data in config (USE_TRIP_LEVEL=True) to reduce overfitting.")


if __name__ == "__main__":
    main()
