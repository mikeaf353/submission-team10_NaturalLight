# Boston Kidney Biopsy Cohort — Hackathon Starter Code

## Task

Predict **acute tubular injury (ATI)** from plasma proteomics, as a binary classification problem:

| Label | Meaning | Source |
|-------|---------|--------|
| **0** | No ATI | biopsy `ati` score = 1 (None) |
| **1** | ATI present | biopsy `ati` score = 2 / 3 / 4 (Mild / Moderate / Severe) |

Predictions will be evaluated on an **external held-out validation cohort** not provided here.  Your model should generalise beyond the BKBC training data.

---

## Data

The data file you will work with is:

```
../merged_anonymized.csv
```

It contains **434 patients** (one row each) with the following columns:

| Column group | Description |
|---|---|
| `s_id` | Anonymised patient identifier |
| `age`, `sex` | Demographic covariates |
| `baseline_egfr_23` | Baseline eGFR (ml/min/1.73 m²) |
| `ati` | ATI severity score (1–4); your prediction target |
| `feature_XXXX` × 6,592 | Log₂-normalised, ANML-normalised SomaScan plasma protein abundances |

> **Protein mapping:** columns are anonymised. To recover gene symbols, join
> `../anonymization_key.csv` with `../soma_seqid_mapping.csv`.

---

## Environment setup

**Recommended — uv (much faster than pip):**

```bash
# Install uv (one-time)
curl -Lsf https://astral.sh/uv/install.sh | sh

# Create venv and install
cd /projectnb/medaihack/BKBC-hackathon
uv venv
source .venv/bin/activate
uv pip install -r starter_code/requirements.txt
```

**Fallback — standard pip:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r starter_code/requirements.txt
```

**Verify the installation:**

```bash
python starter_code/model.py
```

---

## Pipeline

```
merged_anonymized.csv
        │
        ▼
  preprocess.py
  ├── load + clean data
  ├── build feature matrix (6,592 protein + 3 clinical)
  ├── stratified 80/20 train/test split
  └── save splits to disk
        │   splits/train.npz
        │   splits/test.npz
        │   splits/feature_cols.json
        ▼
  train_eval.py
  ├── 5-fold CV on training set  →  CV metrics + confusion matrix
  ├── train final model on full training set
  ├── evaluate on held-out test set
  └── save XGBoost model + feature importance
        │   results/xgboost_model.json
        │   results/feature_cols.json
        │   results/xgboost_feature_importance.{png,csv}
        │   results/ati_cv_confusion_matrix_{model}.png
        ▼
  predict.py  (run on any new CSV with the same feature columns)
        └── predictions.csv
```

---

## Step 1 — Preprocess

```bash
python starter_code/preprocess.py
# or with explicit paths:
python starter_code/preprocess.py \
    --data /path/to/merged_anonymized.csv \
    --out  /path/to/splits/
```

Reads `merged_anonymized.csv`, extracts features and binary ATI labels, and writes stratified train/test splits to disk.  This only needs to run once.

---

## Step 2 — Train and evaluate

```bash
python starter_code/train_eval.py
# or with explicit paths:
python starter_code/train_eval.py \
    --splits /path/to/splits/ \
    --out    /path/to/results/
```

Runs 5-fold cross-validation on the training partition, trains both XGBoost and Lasso Logistic Regression, evaluates each on the held-out test set, and saves the XGBoost model for later prediction.

**Expected console output (approximate):**

```
============================================================
MODEL: XGBoost
============================================================
=== BKBC Train — 5-Fold CV ===
              precision    recall  f1-score   support
      No ATI       0.72      0.74      0.73       182
         ATI       0.70      0.68      0.69       159
AUC (ROC):  0.762
Log loss:   0.591

=== BKBC Held-Out Test Set ===
...
```

**Saved outputs:**

```
results/
├── ati_cv_confusion_matrix_xgboost.png
├── ati_cv_confusion_matrix_lasso_lr.png
├── xgboost_feature_importance.png
├── xgboost_feature_importance.csv
├── xgboost_model.json
└── feature_cols.json
```

---

## Step 3 — Predict on new data

Once a model is trained, `predict.py` applies it to any CSV that shares the same feature schema:

```bash
python starter_code/predict.py --data /path/to/new_cohort.csv

# Full options:
python starter_code/predict.py \
    --data     /path/to/new_cohort.csv \
    --model    /path/to/results/xgboost_model.json \
    --features /path/to/results/feature_cols.json \
    --out      predictions.csv
```

If the new file contains an `ati` column, evaluation metrics are printed automatically.

**Output columns:**

| Column | Description |
|--------|-------------|
| `sample_id` | Patient identifier (`s_id` if present, else row index) |
| `prob_ati` | Predicted probability of ATI (0–1) |
| `pred_label` | Hard prediction: 0 = No ATI, 1 = ATI |
| `true_label` | Ground truth (only if `ati` column is present) |

---

## File structure

```
starter_code/
├── README.md          ← this file
├── requirements.txt   ← Python dependencies (see setup above)
├── model.py           ← model definitions and shared constants
├── preprocess.py      ← data loading, feature extraction, train/test split
├── train_eval.py      ← cross-validation, training, internal evaluation
├── predict.py         ← inference on new / external data
└── run_example.sh     ← SGE job script template
```

After running the pipeline, the parent directory will contain:

```
BKBC-hackathon/
├── merged_anonymized.csv     ← input data (do not modify)
├── anonymization_key.csv     ← feature_XXXX → original SeqId
├── soma_seqid_mapping.csv    ← SeqId → gene symbol / UniProt
├── splits/                   ← generated by preprocess.py
│   ├── train.npz
│   ├── test.npz
│   └── feature_cols.json
└── results/                  ← generated by train_eval.py
    ├── xgboost_model.json
    ├── feature_cols.json
    ├── xgboost_feature_importance.{png,csv}
    └── ati_cv_confusion_matrix_{model}.png
```

---

## Tips for improving beyond the baseline

1. **Feature selection** — 6,592 features is high-dimensional.  Use LASSO coefficients, univariate tests, or biological priors to select a focused subset that may generalise better.

2. **Hyperparameter tuning** — try `optuna` or `GridSearchCV` on `n_estimators`, `max_depth`, `colsample_bytree`, or the LR regularisation strength `C`.

3. **Alternative models** — random forests, ElasticNet, neural networks, or ensemble stacking.

4. **Clinical-only baseline** — train on `age`, `sex`, `baseline_egfr_23` alone to quantify the protein contribution.

5. **Class imbalance** — experiment with `scale_pos_weight` in XGBoost or `class_weight='balanced'` in scikit-learn.

6. **De-anonymise features** — use `anonymization_key.csv` + `soma_seqid_mapping.csv` to recover gene symbols; incorporate domain knowledge about kidney biology.

7. **Cross-cohort robustness** — the external validation cohort uses a slightly different SomaScan normalisation (ANML+SMP vs. ANML).  Quantile normalisation or z-score alignment across cohorts may improve transferability.

---

## Getting help

```bash
python starter_code/preprocess.py --help
python starter_code/train_eval.py --help
python starter_code/predict.py    --help
```
