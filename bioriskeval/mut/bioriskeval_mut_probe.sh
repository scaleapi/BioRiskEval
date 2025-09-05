#!/usr/bin/env bash
set -euo pipefail

# Step 0: Follow BioNemo instructions to download Evo2 7B and 40B, ESM2 650M and 3B. Get Evo2 7B finetuned checkpoints here [TODO]


# Step 1: Extract representations andCreate the probe dataset for train and val splits  
MODEL_SIZE="7b_arc_longcontext"

LAYER_NAMES=(
  "decoder.layers.0"
  "decoder.layers.1"
  "decoder.layers.2"
  "decoder.layers.3"
  "decoder.layers.4"
  "decoder.layers.5"
  "decoder.layers.6"
  "decoder.layers.7"
  "decoder.layers.8"
  "decoder.layers.9"
  "decoder.layers.10"
  "decoder.layers.11"
  "decoder.layers.12"
  "decoder.layers.13"
  "decoder.layers.14"
  "decoder.layers.15"
  "decoder.layers.16"
  "decoder.layers.17"
  "decoder.layers.18"
  "decoder.layers.19"
  "decoder.layers.20"
  "decoder.layers.21"
  "decoder.layers.22"
  "decoder.layers.23"
  "decoder.layers.24"
  "decoder.layers.25"
  "decoder.layers.26"
  "decoder.layers.27"
  "decoder.layers.28"
  "decoder.layers.29"
  "decoder.layers.30"
  "decoder.layers.31"
)


echo "=== DMS Probe Dataset Creation (multiple checkpoints) ==="

# List of checkpoint paths to iterate over
CHECKPOINT_PATHS=(
  "/workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m"
)

for i in "${!CHECKPOINT_PATHS[@]}"; do
  CHECKPOINT_PATH="${CHECKPOINT_PATHS[$i]}"
  CUDA_VISIBLE_DEVICES="0,1,2,3" python /workspaces/BioRiskEval/bioriskeval/mut/create_dms_probe_dataset.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_size "${MODEL_SIZE}" \
    --layer_names "${LAYER_NAMES[@]}" \
    --k 500 \
    --batch_size 128 \
    --seed 42 \
    --tensor_parallel_size 4 
done


# Step 2: Fit probes for all layers, and save the best probe for each layer based on train RMSE and val Spearman correlation

PROBES_ROOT="/workspaces/BioRiskEval/bioriskeval/mut/saved_probes_stratified/k=500_seed=42"
DATASETS_ROOT="/workspaces/BioRiskEval/bioriskeval/mut/data/DMS_Probe"

python /workspaces/BioRiskEval/bioriskeval/mut/sweep_dms_probe.py \
    --task_type continuous \
    --closed_form \
    --gpu 0 \
    --save_best_by_train_rmse \
    --save_best_by_val_spearman \
    --results_dir /workspaces/BioRiskEval/bioriskeval/mut/probe_results_stratified/k=500_seed=42 \
    --probes_root $PROBES_ROOT \
    --datasets_root $DATASETS_ROOT


# Step 3: Save the probe dataset for the test set 

echo "=== DMS Probe Dataset Creation (test-only for selected layers from saved_probes) ==="

# Test-only extraction for 7B checkpoints based on selected layers
MODEL_SIZE="7b_arc_longcontext"
for i in "${!CHECKPOINT_PATHS[@]}"; do
  CHECKPOINT_PATH="${CHECKPOINT_PATHS[$i]}"
  echo ">>> Resolving layers for checkpoint: ${CHECKPOINT_PATH}"
  LAYER_NAMES_STR=$(python /workspaces/BioRiskEval/bioriskeval/mut/probe_layer_utils.py \
    --probes_root "${PROBES_ROOT}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --format bash)
  read -r -a DYN_LAYER_NAMES <<< "${LAYER_NAMES_STR}"
  if [ ${#DYN_LAYER_NAMES[@]} -eq 0 ]; then
    echo "[WARN] No layers found for ${CHECKPOINT_PATH}; skipping test-only extraction"
    continue
  fi
  echo ">>> Extracting test-only for layers: ${DYN_LAYER_NAMES[*]}"
  CUDA_VISIBLE_DEVICES="0,1,2,3" python /workspaces/BioRiskEval/bioriskeval/mut/create_dms_probe_dataset.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_size "${MODEL_SIZE}" \
    --layer_names "${DYN_LAYER_NAMES[@]}" \
    --k 500 \
    --batch_size 128 \
    --seed 42 \
    --tensor_parallel_size 4 \
    --save_test
done


# Step 4: Evaluate the best probe on the test set
python /workspaces/BioRiskEval/bioriskeval/mut/test_dms_probe.py \
  --results_dir probe_results_stratified/k=500_seed=42/closed_form \
  --datasets_root $DATASETS_ROOT \
  --probes_root $PROBES_ROOT \
  --eval_best_by_train_rmse \
  --eval_best_by_val_spearman


