#!/bin/bash

# Zero-shot/logprob Evaluation 


# Step 0: Follow BioNemo instructions to download Evo2 7B and 40B, ESM2 650M and 3B. Get Evo2 7B finetuned checkpoints here [TODO]


# Step 1. Process the DMS dataset to generate nucleotide sequences or download the pre-processed dataset

original_data_dir="/workspaces/BioRiskEval/bioriskeval/mut/data/DMS_ProteinGym_substitutions"
target_data_dir="/workspaces/BioRiskEval/bioriskeval/mut/data/DMS_ProteinGym_substitutions/nucleotides"

python /workspaces/BioRiskEval/bioriskeval/mut/nucleotide_data_pipeline.py \
        --input_folder $original_data_dir \
        --input_file $original_data_dir/DMS_substitutions.csv \
        --output_dir $target_data_dir \
        --seed 42 \
        --force_blast


#OR download the processed dataset here: [TODO]



# Step 2: Evaluate the logprob for 7B and 40B Evo2 checkpoints 

DEST_DIR="/workspaces/BioRiskEval/attack/checkpoints/"
MODEL_SIZE="7b_arc_longcontext"  

# Define relative checkpoint paths
CHECKPOINTS=(
    "orig_checkpoints/nemo2_evo2_7b_1m"

)

run_evaluation() {
    local checkpoint_path="$1"
    local gpu_devices="$2"
    local model_dir=$(echo "$checkpoint_path" | cut -d'/' -f1)
    local tensor_parallel_size="$3"
    
    echo "Starting evaluation of $model_dir on GPUs $gpu_devices..."
    
    CUDA_VISIBLE_DEVICES="$gpu_devices" python eval/eval_fitness.py \
        --ckpt-dir "$DEST_DIR$checkpoint_path" \
        --model-size "$MODEL_SIZE" \
        --batch-size 128 \
        --tensor-parallel-size "$tensor_parallel_size" \
        --DMS_filenames $original_data_dir/virus_reproduction.csv \
        --DMS_scores_folder $target_data_dir
    
    echo "Completed evaluation of $model_dir"
}

echo "Starting 7B checkpoint evaluations..."
for checkpoint_path in "${CHECKPOINTS[@]}"; do
    run_evaluation "$checkpoint_path" "0,1,2,3" 4
done



# Step 3: Evaluate the masked marginal for ESM2 models for comparison 
CHECKPOINTS=(
    "esm2_650m"
    "esm2_3b"
)

run_evaluation() {
    local checkpoint_path="$1"
    local gpu_devices="$2"
    local model_id=$(echo "$checkpoint_path" | cut -d'/' -f1)

    echo "Starting evaluation of $model_id on GPUs $gpu_devices..."

    CUDA_VISIBLE_DEVICES="$gpu_devices" python eval/eval_fitness_esm2.py \
        --ckpt-dir "${DEST_DIR}${checkpoint_path}" \
        --batch-size 64 \
        --tensor-parallel-size 1 \
        --DMS_filenames $original_data_dir/virus_reproduction.csv \
        --DMS_scores_folder $target_data_dir \
        --DMS_reference_file_path $original_data_dir/DMS_substitutions.csv

    echo "Completed evaluation of $model_id"
}

echo "Starting sequential evaluations..."
for checkpoint_path in "${CHECKPOINTS[@]}"; do
    run_evaluation "$checkpoint_path" "0"
done


# Step 4: Evaluate Evo2 and ESM2 on the test split for comparison with probing results 


