#!/bin/bash

# Simple launcher for ESM-2 fitness evaluation in the style of other launch scripts.
# Note:
# - If you use a BioNeMo tag (e.g., esm2/nv_650m:2.1) set DEST_DIR to empty string "".
# - If you use local relative checkpoint directories, set DEST_DIR to your base path.

MODEL_NOTE="esm2"  # for printing/logging only

# Set to base directory for checkpoints; leave empty when using BioNeMo tags in CHECKPOINTS
DEST_DIR=""  # e.g., "/workspaces/BioRiskEval/attack/checkpoints/"

# Define checkpoint identifiers (either relative dirs or BioNeMo tags)
CHECKPOINTS=(
    # Examples:
    # "orig_checkpoints/esm2_650m_checkpoint_dir"
    # "esm2/nv_650m:2.1"
    # "esm2/nv_3b:2.1"
    "/workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/esm2_650m"
    "/workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/esm2_3b"
)

run_evaluation() {
    local checkpoint_path="$1"
    local gpu_devices="$2"
    local model_id=$(echo "$checkpoint_path" | cut -d'/' -f1)

    echo "Starting ${MODEL_NOTE} evaluation of $model_id on GPUs $gpu_devices..."

    CUDA_VISIBLE_DEVICES="$gpu_devices" python eval/eval_fitness_esm2.py \
        --ckpt-dir "${DEST_DIR}${checkpoint_path}" \
        --batch-size 64 \
        --tensor-parallel-size 1 \
        --DMS_filenames /workspaces/BioRiskEval/attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv \
        --DMS_scores_folder /workspaces/BioRiskEval/attack/data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides/ \
        --DMS_reference_file_path /workspaces/BioRiskEval/attack/data/eval_dataset/DMS_ProteinGym_substitutions/DMS_substitutions.csv

    echo "Completed ${MODEL_NOTE} evaluation of $model_id"
}

echo "Starting sequential evaluations..."
for checkpoint_path in "${CHECKPOINTS[@]}"; do
    run_evaluation "$checkpoint_path" "1"
done

# # OPTION: Simple batched parallel (uncomment to use)
# echo "Starting simple batched parallel evaluations..."
# for i in "${!CHECKPOINTS[@]}"; do
#     checkpoint_path="${CHECKPOINTS[$i]}"
#     if (( i % 2 == 0 )); then
#         gpu_devices="0,1,2,3"
#     else
#         gpu_devices="4,5,6,7"
#     fi
#     run_evaluation "$checkpoint_path" "$gpu_devices" &
#     if (( (i + 1) % 2 == 0 )); then
#         wait
#     fi
# done
# wait

# echo "All evaluations completed!"



