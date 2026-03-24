#!/bin/bash
#$ -N bkbc_ati
#$ -P medaihack
#$ -l h_rt=00:30:00      # full pipeline completes in a few minutes
#$ -pe omp 8             # 8 cores (XGBClassifier uses n_jobs=-1)
#$ -l mem_per_core=4G    # 32 GB total RAM
#$ -j y                  # merge stdout and stderr into one log file
#$ -o bkbc_ati.log

# ── Activate virtual environment ──────────────────────────────────────────────
# Update this path if your venv lives elsewhere.
source /projectnb/medaihack/BKBC-hackathon/.venv/bin/activate

DATA=/projectnb/medaihack/BKBC-hackathon/merged_anonymized.csv

# ── Step 1: Build train/test splits ───────────────────────────────────────────
# Reads merged_anonymized.csv, extracts features + labels, saves splits to disk.
# Only needs to run once; re-run if you change CLINICAL_FEATURES or TEST_SIZE.

python ./preprocess.py \
    --data "$DATA" \
    --out  /projectnb/medaihack/BKBC-hackathon/splits

# ── Step 2: Train and evaluate ────────────────────────────────────────────────
# Runs 5-fold CV on training set, trains final models, evaluates on held-out
# test set, saves XGBoost model + feature importance outputs.

python ./train_eval.py \
    --splits /projectnb/medaihack/BKBC-hackathon/splits \
    --out    /projectnb/medaihack/BKBC-hackathon/results

# ── Step 3 (optional): Predict on a new cohort ────────────────────────────────
# Uncomment and set --data to your external CSV once the model is trained.
# If the CSV contains an 'ati' column, evaluation metrics are printed automatically.

# python starter_code/predict.py \
#     --data     /path/to/new_cohort.csv \
#     --model    /projectnb/medaihack/BKBC-hackathon/results/xgboost_model.json \
#     --features /projectnb/medaihack/BKBC-hackathon/results/feature_cols.json \
#     --out      predictions.csv
