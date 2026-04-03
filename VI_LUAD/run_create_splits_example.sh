#!/bin/bash

# run_create_splits_example.sh
# ----------------------------
# Creates 5-fold stratified patient-level cross-validation splits from the
# clinical label file. Run this once before training.
#
# Usage (from your project directory in an interactive session):
#   bash starter_code/run_create_splits_example.sh
#
# This takes only a few seconds and requires no GPU.

module load miniconda
conda activate vi_luad

# Splits are saved to starter_code/splits/ inside your project directory.
# Make sure you run this script from <your_project_dir>.
python starter_code/create_splits.py \
    --label_file /projectnb/medaihack/VI_LUAD_Project/Clinical_Data/hackathon_label.txt \
    --splits_dir <your_project_dir>/starter_code/splits

# Optional: use a different number of folds or random seed
#   --n_folds    3
#   --random_seed 0
