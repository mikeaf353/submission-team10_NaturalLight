#!/usr/bin/env python3
"""
predict.py — Run a trained model on new / external data
=========================================================
Loads the XGBoost model saved by train_eval.py and generates predictions
for any new dataset that shares the same feature schema.

Column order in the input file does not matter — alignment to the training
feature list is handled automatically.  Missing columns are filled with NaN
(XGBoost handles them natively via its built-in missing-value support).

USAGE
-----
    # Minimal — uses default model / feature paths from ../results/
    python predict.py --data /path/to/new_cohort.csv

    # Full control
    python predict.py \\
        --data     /path/to/new_cohort.csv \\
        --model    /path/to/results/xgboost_model.json \\
        --features /path/to/results/feature_cols.json \\
        --out      predictions.csv

OUTPUT COLUMNS
--------------
    sample_id   — s_id if present in the input file, otherwise row index
    prob_ati    — predicted probability of ATI (0–1)
    pred_label  — hard prediction: 0 = No ATI, 1 = ATI  (threshold 0.5)
    true_label  — ground-truth (0/1) if an 'ati' column is present; else omitted

PYTHON API EXAMPLE
------------------
    from predict import load_model, prepare_features, run_predict
    import pandas as pd

    model, feature_cols = load_model(
        model_path    = "../results/xgboost_model.json",
        features_path = "../results/feature_cols.json",
    )
    df     = pd.read_csv("new_cohort.csv", low_memory=False, na_values=[".", ""])
    X, ids = prepare_features(df, feature_cols)
    out    = run_predict(model, X, ids)
    print(out.head())
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR       = Path(__file__).resolve().parent
_DEFAULT_MODEL    = _SCRIPT_DIR.parent / "results" / "xgboost_model.json"
_DEFAULT_FEATURES = _SCRIPT_DIR.parent / "results" / "feature_cols.json"


# ── SECTION 1: CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained XGBoost ATI model on new proteomics data."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to new data CSV (same format as merged_anonymized.csv)",
    )
    p.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help="Path to xgboost_model.json (default: ../results/xgboost_model.json)",
    )
    p.add_argument(
        "--features",
        default=str(_DEFAULT_FEATURES),
        help="Path to feature_cols.json  (default: ../results/feature_cols.json)",
    )
    p.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    return p.parse_args()


# ── SECTION 2: Load model ──────────────────────────────────────────────────────

def load_model(model_path: str, features_path: str):
    """
    Restore a trained XGBoost model and its feature column list from disk.

    The feature list records the exact column order used at training time.
    This is essential — XGBoost uses positional feature indices internally,
    not column names.

    Returns
    -------
    model        : fitted XGBClassifier
    feature_cols : list[str]  ordered column names expected by the model
    """
    logging.info(f"Loading model        : {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)

    logging.info(f"Loading feature list : {features_path}")
    with open(features_path) as f:
        feature_cols = json.load(f)

    logging.info(f"Model expects {len(feature_cols)} features")
    return model, feature_cols


# ── SECTION 3: Prepare input data ─────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, feature_cols: list):
    """
    Align a new DataFrame to the feature columns expected by the model.

    Parameters
    ----------
    df           : raw input DataFrame (any column order)
    feature_cols : ordered list loaded from feature_cols.json

    Returns
    -------
    X   : np.ndarray  (n_samples, n_features)
    ids : pd.Series   sample identifiers
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.warning(
            f"{len(missing)} feature columns absent from input — filled with NaN. "
            f"First few: {missing[:5]}"
        )

    X    = df.reindex(columns=feature_cols).values.astype(float)
    ids  = df["s_id"] if "s_id" in df.columns else pd.RangeIndex(len(df))
    return X, ids


# ── SECTION 4: Predict ─────────────────────────────────────────────────────────

def run_predict(model, X: np.ndarray, ids, y_true=None) -> pd.DataFrame:
    """
    Generate ATI predictions for a feature matrix.

    Parameters
    ----------
    model  : fitted XGBClassifier
    X      : feature matrix (n_samples, n_features)
    ids    : sample identifiers (Series or array-like)
    y_true : optional ground-truth labels (0/1) for evaluation

    Returns
    -------
    pd.DataFrame with columns: sample_id, prob_ati, pred_label [, true_label]
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    out = pd.DataFrame({
        "sample_id" : ids,
        "prob_ati"  : y_prob,
        "pred_label": y_pred,
    })
    if y_true is not None:
        out["true_label"] = np.asarray(y_true)
    return out


# ── SECTION 5: Evaluation ──────────────────────────────────────────────────────

def evaluate(results: pd.DataFrame):
    """Print classification metrics when ground-truth labels are available."""
    from sklearn.metrics import classification_report, roc_auc_score, log_loss

    y_true = results["true_label"].values
    y_pred = results["pred_label"].values
    y_prob = results["prob_ati"].values

    print(f"\n{'=' * 50}")
    print(f"  Samples : {len(results)}  |  "
          f"No ATI: {(y_pred==0).sum()}  |  ATI: {(y_pred==1).sum()}")
    print(f"{'=' * 50}")

    if len(np.unique(y_true)) > 1:
        print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
        print(f"AUC (ROC) : {roc_auc_score(y_true, y_prob):.3f}")
        print(f"Log loss  : {log_loss(y_true, y_prob):.3f}")
    else:
        print("  (Only one class present in labels — AUC not defined)")


# ── SECTION 6: Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model, feature_cols = load_model(args.model, args.features)

    logging.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples loaded")

    X, ids = prepare_features(df, feature_cols)

    # Extract ground-truth labels if available
    y_true = None
    if "ati" in df.columns:
        y_true = (pd.to_numeric(df["ati"], errors="coerce") > 1).astype("Int64")
        logging.info("Found 'ati' column — will compute evaluation metrics")

    results = run_predict(model, X, ids, y_true=y_true)

    if y_true is not None:
        evaluate(results)

    results.to_csv(args.out, index=False)
    logging.info(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
