#!/bin/bash
# This script is used to preprocess the data used to fine-tuning.


csv_path="/workspaces/BioRiskEval/attack/data/eval_dataset/gammapapillomavirus.csv"
# Step1: Convert csv to fna

python convert_csv_to_fna.py --file_name $csv_path






