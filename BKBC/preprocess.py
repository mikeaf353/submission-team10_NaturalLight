#!/usr/bin/env python3
"""
preprocess.py — Data loading and train/test split
==================================================
Reads merged_anonymized.csv, builds the feature matrix and binary ATI labels,
performs a stratified train/test split, and saves the two partitions to disk
as compressed NumPy archives.

This script is a one-time step.  train_eval.py and predict.py load the
outputs rather than re-reading the raw CSV every time.

LABELS
------
Binary ATI (derived from biopsy histopathology):
    ati == 1  (None)             → label 0  (no ATI)
    ati == 2 / 3 / 4             → label 1  (ATI present)
    (Mild / Moderate / Severe)

FEATURES
--------
    Proteomics : 6,592 log₂-normalised, ANML-normalised SomaScan abundances
                 (anonymised as feature_XXXX columns)
    Clinical   : age, sex, baseline_egfr_23  (see model.CLINICAL_FEATURES)

USAGE
-----
    python preprocess.py
    python preprocess.py --data /path/to/merged_anonymized.csv \\
                         --out  /path/to/splits/

OUTPUTS  (written to --out directory)
-------
    train.npz          — X_train, y_train arrays
    test.npz           — X_test,  y_test  arrays
    feature_cols.json  — ordered list of feature column names
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import CLINICAL_FEATURES, TEST_SIZE, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR   = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "merged_anonymized.csv"
_DEFAULT_OUT  = _SCRIPT_DIR.parent / "splits"


# ── SECTION 1: CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build train/test splits from merged_anonymized.csv."
    )
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to merged_anonymized.csv (default: ../merged_anonymized.csv)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Directory to write split files (default: ../splits/)",
    )
    return p.parse_args()


# ── SECTION 2: Data loading ────────────────────────────────────────────────────

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the pre-merged anonymised dataset.

    '.' is Stata's missing-value sentinel and is translated to NaN on load.
    low_memory=False prevents DtypeWarning on mixed-type columns.
    """
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples, {len(df.columns)} columns")
    return df


# ── SECTION 3: Feature and label extraction ────────────────────────────────────

def build_features_and_labels(df: pd.DataFrame):
    """
    Extract the feature matrix X and binary ATI label vector y.

    Feature set = anonymised protein columns (feature_XXXX) + CLINICAL_FEATURES.
    Rows missing any feature or the ATI label are dropped (complete-case analysis).

    Parameters
    ----------
    df : raw DataFrame from load_data()

    Returns
    -------
    X            : np.ndarray  (n_samples, n_features)
    y            : np.ndarray  (n_samples,)  0 = No ATI, 1 = ATI
    feature_cols : list[str]   column names in the same order as X columns
    """
    protein_cols = [c for c in df.columns if c.startswith("feature_")]
    feature_cols = protein_cols + CLINICAL_FEATURES

    required = feature_cols + ["ati"]
    df_clean = df[required].copy().dropna(subset=required)

    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        logging.warning(f"Dropped {n_dropped} rows with missing values")

    y = (df_clean["ati"].astype(int) > 1).astype(int)   # 0 = No ATI, 1 = ATI
    X = df_clean[feature_cols].values

    logging.info(
        f"Samples: {len(df_clean)} | No ATI: {(y == 0).sum()} | ATI: {(y == 1).sum()}"
    )
    logging.info(
        f"Features: {len(protein_cols)} protein + {len(CLINICAL_FEATURES)} clinical"
        f" = {len(feature_cols)} total"
    )
    return X, y, feature_cols


# ── SECTION 4: Train / test split ─────────────────────────────────────────────

def split_data(X: np.ndarray, y: np.ndarray):
    """
    Stratified 80 / 20 train-test split.

    The test set is held out and must not be used during cross-validation or
    hyperparameter tuning.  It represents the internal held-out evaluation
    set — treat it as unseen data until final reporting.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_SEED,
    )
    logging.info(
        f"Train: {len(y_train)} samples (No ATI: {(y_train==0).sum()}, ATI: {(y_train==1).sum()})"
    )
    logging.info(
        f"Test : {len(y_test)}  samples (No ATI: {(y_test==0).sum()},  ATI: {(y_test==1).sum()})"
    )
    return X_train, X_test, y_train, y_test


# ── SECTION 5: Save splits ─────────────────────────────────────────────────────

def save_splits(
    X_train, y_train,
    X_test,  y_test,
    feature_cols: list,
    out_dir: Path,
):
    """
    Persist the train and test partitions to compressed NumPy archives.

    Files written:
        out_dir/train.npz         — X_train, y_train
        out_dir/test.npz          — X_test,  y_test
        out_dir/feature_cols.json — ordered column name list
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "test.npz",  X=X_test,  y=y_test)

    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    logging.info(f"Saved: {out_dir / 'train.npz'}")
    logging.info(f"Saved: {out_dir / 'test.npz'}")
    logging.info(f"Saved: {out_dir / 'feature_cols.json'}")


# ── SECTION 6: Main ────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out)

    df = load_data(args.data)
    X, y, feature_cols = build_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_splits(X_train, y_train, X_test, y_test, feature_cols, out_dir)

    print(f"\nSplits saved to {out_dir}/")
    print("Next step: python train_eval.py")


if __name__ == "__main__":
    main()
