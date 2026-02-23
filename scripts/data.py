"""
Data loading and preprocessing for driver sensor data.
Supports raw row-level and trip-level aggregation (one row per one-way FWD/BWD).
FWD = source → destination, BWD = destination → source.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import (
    DATA_PATH,
    FEATURE_COLS_NUM,
    FEATURE_COLS_CAT,
    WINDOW_SIZE,
    SAMPLE_SIZE,
    RANDOM_STATE,
    USE_TRIP_LEVEL,
    SPLIT_BY_DRIVER,
    TRIP_SEQ_LENGTH,
)


def _build_trip_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row with a trip ID. One trip = one contiguous one-way segment
    (same DriverID, Date, RouteDirection until direction changes).
    FWD = source→destination, BWD = destination→source.
    """
    df = df.sort_values(["DriverID", "Date", "Timestamp"]).reset_index(drop=True)

    def segment_trips(g):
        ch = (g["RouteDirection"] != g["RouteDirection"].shift()).astype(int)
        ch.iloc[0] = 1
        return ch.cumsum()

    df["_trip_seg"] = df.groupby(["DriverID", "Date"], group_keys=False).apply(segment_trips)
    df["trip_uid"] = (
        df["DriverID"].astype(str)
        + "_"
        + df["Date"].astype(str)
        + "_"
        + df["_trip_seg"].astype(str)
    )
    return df.drop(columns=["_trip_seg"], errors="ignore")


def _aggregate_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate each trip (one-way FWD or BWD) to one row: mean of numeric features,
    mode of StressLabel and ActivityState. Keeps DriverID, Date, RouteDirection.
    """
    num_cols = [c for c in FEATURE_COLS_NUM if c in df.columns]
    agg_dict = {c: "mean" for c in num_cols}
    agg_dict["StressLabel"] = lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]
    agg_dict["ActivityState"] = lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]
    agg_dict["RouteDirection"] = "first"
    agg_dict["DriverID"] = "first"
    agg_dict["Date"] = "first"

    trip_df = df.groupby("trip_uid", as_index=False).agg(agg_dict)
    return trip_df


def _split_by_driver(df: pd.DataFrame, drivers_col: str = "DriverID", test_size: float = 0.2):
    """Split so that train and val have disjoint sets of drivers (no leakage)."""
    drivers = df[drivers_col].unique()
    n = len(drivers)
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n)
    n_val = max(1, int(n * test_size))
    val_drivers = set(drivers[perm[:n_val]])
    train_drivers = set(drivers[perm[n_val:]])
    train_mask = df[drivers_col].isin(train_drivers).values
    val_mask = df[drivers_col].isin(val_drivers).values
    return train_mask, val_mask, train_drivers, val_drivers


def load_and_preprocess():
    """
    Load CSV, preprocess, and return train/val splits.
    - If USE_TRIP_LEVEL: one row per one-way trip (FWD/BWD); sequences = windows of trips per driver.
    - If SPLIT_BY_DRIVER: train/val split by driver so validation is on unseen drivers.
    """
    df = pd.read_csv(DATA_PATH)
    if SAMPLE_SIZE:
        df = df.sample(
            n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE
        ).reset_index(drop=True)

    if USE_TRIP_LEVEL:
        df = _build_trip_ids(df)
        df = _aggregate_trips(df)
        df = df.sort_values(["DriverID", "Date", "trip_uid"]).reset_index(drop=True)
    else:
        df = df.sort_values(["DriverID", "Timestamp"]).reset_index(drop=True)

    le_activity = LabelEncoder()
    le_route = LabelEncoder()
    le_label = LabelEncoder()
    df["ActivityState_enc"] = le_activity.fit_transform(df["ActivityState"].astype(str))
    df["RouteDirection_enc"] = le_route.fit_transform(df["RouteDirection"].astype(str))
    y = le_label.fit_transform(df["StressLabel"])
    class_names = list(le_label.classes_)
    num_classes = len(class_names)

    num_cols = FEATURE_COLS_NUM
    cat_cols = FEATURE_COLS_CAT
    feature_cols = num_cols + cat_cols

    X_num = df[num_cols].values.astype(np.float32)
    X_cat = df[cat_cols].values.astype(np.float32)
    X_full = np.hstack([X_num, X_cat])
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    if SPLIT_BY_DRIVER:
        train_mask, val_mask, train_drivers, val_drivers = _split_by_driver(df, "DriverID", 0.2)
        idx_train = np.where(train_mask)[0]
        idx_val = np.where(val_mask)[0]
    else:
        idx_all = np.arange(len(df))
        idx_train, idx_val = train_test_split(
            idx_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        train_drivers = val_drivers = set()

    if USE_TRIP_LEVEL:
        # Tabular: one row per trip
        X_tab_tr, X_tab_val = X_full[idx_train], X_full[idx_val]
        y_tab_tr, y_tab_val = y[idx_train], y[idx_val]

        # Sequences: windows of TRIP_SEQ_LENGTH consecutive trips per driver
        seqs, seq_labels, seq_drivers = [], [], []
        for driver_id, g in df.groupby("DriverID"):
            g = g.reset_index(drop=True)
            feats = X_full[g.index]
            labs = y[g.index]
            for i in range(len(g) - TRIP_SEQ_LENGTH):
                seqs.append(feats[i : i + TRIP_SEQ_LENGTH])
                seq_labels.append(labs[i + TRIP_SEQ_LENGTH - 1])
                seq_drivers.append(driver_id)

        if seqs:
            X_seq = np.array(seqs, dtype=np.float32)
            y_seq = np.array(seq_labels)
            seq_drivers = np.array(seq_drivers)
            seq_train_mask = np.array([d in train_drivers for d in seq_drivers])
            seq_val_mask = np.array([d in val_drivers for d in seq_drivers])
            X_seq_tr = X_seq[seq_train_mask]
            X_seq_val = X_seq[seq_val_mask]
            y_seq_tr = y_seq[seq_train_mask]
            y_seq_val = y_seq[seq_val_mask]
        else:
            X_seq_tr = np.zeros((0, TRIP_SEQ_LENGTH, X_full.shape[1]), dtype=np.float32)
            X_seq_val = np.zeros((0, TRIP_SEQ_LENGTH, X_full.shape[1]), dtype=np.float32)
            y_seq_tr = np.array([], dtype=np.int64)
            y_seq_val = np.array([], dtype=np.int64)
        X_seq_full = X_seq if seqs else np.zeros((0, TRIP_SEQ_LENGTH, X_full.shape[1]), dtype=np.float32)
        window_size = TRIP_SEQ_LENGTH
    else:
        # Row-level sequences (original)
        seqs, seq_labels = [], []
        last_row_idx = []
        for _, g in df.groupby("DriverID"):
            inds = g.index.tolist()
            feats = X_full[inds]
            labs = y[inds]
            for i in range(len(g) - WINDOW_SIZE):
                seqs.append(feats[i : i + WINDOW_SIZE])
                seq_labels.append(labs[i + WINDOW_SIZE - 1])
                last_row_idx.append(inds[i + WINDOW_SIZE - 1])

        X_seq = np.array(seqs, dtype=np.float32)
        y_seq = np.array(seq_labels)
        X_tab = X_seq[:, -1, :].copy()
        y_tab = y_seq.copy()
        last_row_idx = np.array(last_row_idx)

        if SPLIT_BY_DRIVER:
            seq_train_mask = train_mask[last_row_idx]
            seq_val_mask = val_mask[last_row_idx]
            X_seq_tr, X_seq_val = X_seq[seq_train_mask], X_seq[seq_val_mask]
            y_seq_tr, y_seq_val = y_seq[seq_train_mask], y_seq[seq_val_mask]
            X_tab_tr, X_tab_val = X_tab[seq_train_mask], X_tab[seq_val_mask]
            y_tab_tr, y_tab_val = y_tab[seq_train_mask], y_tab[seq_val_mask]
        else:
            X_seq_tr, X_seq_val, y_seq_tr, y_seq_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=RANDOM_STATE, stratify=y_seq
            )
            X_tab_tr, X_tab_val, y_tab_tr, y_tab_val = train_test_split(
                X_tab, y_tab, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tab
            )
        X_seq_full = X_seq
        window_size = WINDOW_SIZE

    preprocess = {
        "scaler": scaler,
        "le_label": le_label,
        "le_activity": le_activity,
        "le_route": le_route,
        "feature_cols": feature_cols,
        "window_size": window_size,
    }

    return {
        "X_seq_tr": X_seq_tr,
        "X_seq_val": X_seq_val,
        "y_seq_tr": y_seq_tr,
        "y_seq_val": y_seq_val,
        "X_tab_tr": X_tab_tr,
        "X_tab_val": X_tab_val,
        "y_tab_tr": y_tab_tr,
        "y_tab_val": y_tab_val,
        "X_seq_full": X_seq_full,
        "num_classes": num_classes,
        "class_names": class_names,
        "preprocess": preprocess,
    }
