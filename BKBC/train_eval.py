#!/usr/bin/env python3
"""
train_eval.py — Cross-validation, training, and internal evaluation
====================================================================
Loads the train/test splits produced by preprocess.py, runs stratified
cross-validation on the training set, trains a final model on the full
training set, and evaluates on the held-out test set.

Both XGBoost and Lasso LR are evaluated with the same protocol so their
performance is directly comparable.  The fitted XGBoost model is saved to
disk for later use with predict.py.

USAGE
-----
    python train_eval.py
    python train_eval.py --splits /path/to/splits/ --out /path/to/results/

OUTPUTS  (written to --out directory)
-------
    ati_cv_confusion_matrix_{model}.png  — 5-fold CV confusion matrix per model
    xgboost_feature_importance.png       — top-20 proteins by XGBoost gain
    xgboost_feature_importance.csv       — full ranked feature importance table
    xgboost_model.json                   — trained XGBoost model (→ predict.py)
    feature_cols.json                    — feature column list (→ predict.py)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    log_loss,
)

from model import MODELS, CV_FOLDS, RANDOM_SEED, build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR    = Path(__file__).resolve().parent
_DEFAULT_SPLITS = _SCRIPT_DIR.parent / "splits"
_DEFAULT_OUT    = _SCRIPT_DIR.parent / "results"


# ── SECTION 1: CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train and evaluate ATI classifiers using pre-built splits."
    )
    p.add_argument(
        "--splits",
        default=str(_DEFAULT_SPLITS),
        help="Directory containing train.npz / test.npz / feature_cols.json "
             "(default: ../splits/)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Directory to write model outputs and plots (default: ../results/)",
    )
    return p.parse_args()


# ── SECTION 2: Load splits ─────────────────────────────────────────────────────

def load_splits(splits_dir: Path):
    """
    Load the train and test partitions saved by preprocess.py.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    feature_cols                      : list[str]
    """
    logging.info(f"Loading splits from {splits_dir}...")

    train = np.load(splits_dir / "train.npz")
    test  = np.load(splits_dir / "test.npz")

    with open(splits_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)

    X_train, y_train = train["X"], train["y"]
    X_test,  y_test  = test["X"],  test["y"]

    logging.info(
        f"Train: {len(y_train)} samples | "
        f"No ATI: {(y_train==0).sum()} | ATI: {(y_train==1).sum()}"
    )
    logging.info(
        f"Test : {len(y_test)}  samples | "
        f"No ATI: {(y_test==0).sum()}  | ATI: {(y_test==1).sum()}"
    )
    logging.info(f"Features: {len(feature_cols)}")
    return X_train, y_train, X_test, y_test, feature_cols


# ── SECTION 3: Cross-validation ────────────────────────────────────────────────

def run_cross_validation(model, X_train: np.ndarray, y_train: np.ndarray, name: str):
    """
    Stratified k-fold cross-validation on the training set.

    Uses predict_proba to produce out-of-fold probability estimates, then
    thresholds at 0.5 for hard predictions.  No test-set data is used here.

    Returns
    -------
    y_pred : np.ndarray  hard predictions (0/1)
    y_prob : np.ndarray  P(ATI) probability estimates
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    logging.info(f"[{name}] Running {CV_FOLDS}-fold CV on training set...")
    y_prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


# ── SECTION 4: Training ────────────────────────────────────────────────────────

def train_model(model, X_train: np.ndarray, y_train: np.ndarray, name: str):
    """Fit the model on the full training set."""
    logging.info(f"[{name}] Training on full training set...")
    model.fit(X_train, y_train)
    return model


def predict(model, X: np.ndarray):
    """Return hard predictions and P(ATI) probabilities."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


# ── SECTION 5: Metrics and plots ───────────────────────────────────────────────

def print_metrics(y, y_pred, y_prob, title="Results"):
    print(f"\n=== {title} ===")
    print(classification_report(y, y_pred, target_names=["No ATI", "ATI"]))
    print(f"AUC (ROC):  {roc_auc_score(y, y_prob):.3f}")
    print(f"Log loss:   {log_loss(y, y_prob):.3f}")


def plot_confusion_matrix(y, y_pred, out_path: Path, title=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y, y_pred),
        display_labels=["No ATI", "ATI"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"ATI — {CV_FOLDS}-Fold CV{(' — ' + title) if title else ''}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    logging.info(f"Saved: {out_path}")
    plt.close()


# ── SECTION 6: XGBoost — save model and feature importance ────────────────────

def save_xgboost(model, feature_cols: list, out_dir: Path):
    """
    Persist the trained XGBoost model and feature column list.

    The feature list is required by predict.py to align new input data to
    exactly the same column order used during training.
    """
    model.save_model(str(out_dir / "xgboost_model.json"))
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    logging.info(f"Saved: {out_dir / 'xgboost_model.json'}")
    logging.info(f"Saved: {out_dir / 'feature_cols.json'}")


def plot_feature_importance(model, feature_cols: list, out_dir: Path):
    """Rank features by XGBoost gain, save a bar chart and a CSV."""
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "gain":    model.feature_importances_,
    }).sort_values("gain", ascending=False)

    importance_df.to_csv(out_dir / "xgboost_feature_importance.csv", index=False)
    logging.info(f"Saved: {out_dir / 'xgboost_feature_importance.csv'}")

    top20 = importance_df.head(20)
    print("\nTop 20 features by XGBoost gain:")
    print(top20.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top20["feature"][::-1], top20["gain"][::-1])
    ax.set_xlabel("XGBoost Feature Importance (gain)")
    ax.set_title("Top 20 Features for ATI Prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "xgboost_feature_importance.png", dpi=150)
    logging.info(f"Saved: {out_dir / 'xgboost_feature_importance.png'}")
    plt.close()


# ── SECTION 7: Main ────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    splits_dir = Path(args.splits)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_cols = load_splits(splits_dir)

    fitted_models = {}

    for name in MODELS:
        print("\n" + "=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        model = build_model(name)

        # 1. Cross-validation on training set (test set not touched)
        y_pred_cv, y_prob_cv = run_cross_validation(model, X_train, y_train, name)
        print_metrics(y_train, y_pred_cv, y_prob_cv,
                      title=f"BKBC Train — {CV_FOLDS}-Fold CV")
        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_train, y_pred_cv,
            out_dir / f"ati_cv_confusion_matrix_{slug}.png",
            title=name,
        )

        # 2. Train final model on full training set
        model = build_model(name)   # fresh clone — CV above left model unfitted
        fitted = train_model(model, X_train, y_train, name)
        fitted_models[name] = fitted

        # 3. Evaluate on held-out test set
        y_pred_test, y_prob_test = predict(fitted, X_test)
        print_metrics(y_test, y_pred_test, y_prob_test,
                      title="BKBC Held-Out Test Set")

        # 4. XGBoost-specific outputs
        if name == "XGBoost":
            save_xgboost(fitted, feature_cols, out_dir)
            plot_feature_importance(fitted, feature_cols, out_dir)

    print(f"\nAll outputs written to {out_dir}/")
    print("Next step (external validation): python predict.py --data /path/to/new_data.csv")


if __name__ == "__main__":
    main()
