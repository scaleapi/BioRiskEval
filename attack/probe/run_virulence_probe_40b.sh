#!/bin/bash
export WANDB_API_KEY=<your_wandb_api_key>

PROBE_BATCH=128


for OUTPUT_DIR in "/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/probe_datasets_continuous_40b"; do
for shuffle_labels in "True"; do
CHECKPOINT_PATH="/workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_evo2_40b_1m"

model_name="${CHECKPOINT_PATH##*/}"
MODEL_SIZE="40b_arc_longcontext"
echo "Model name: $model_name"
echo $CHECKPOINT_PATH



python /workspaces/BioRiskEval/attack/probe/create_virulence_probe_dataset_40b.py \
     --checkpoint_path $CHECKPOINT_PATH \
     --model_size $MODEL_SIZE \
     --layer_names decoder.layers.2 decoder.layers.3 decoder.layers.4 decoder.layers.5 decoder.layers.6 decoder.layers.7 decoder.layers.8 decoder.layers.9 decoder.layers.10 decoder.layers.11 decoder.layers.12 decoder.layers.13 decoder.layers.14 decoder.layers.15 decoder.layers.16 decoder.layers.17 decoder.layers.18 decoder.layers.19 decoder.layers.20 decoder.layers.21 decoder.layers.22 decoder.layers.23 decoder.layers.24 decoder.layers.25 decoder.layers.26 decoder.layers.27 decoder.layers.28 decoder.layers.29 decoder.layers.30 decoder.layers.31 decoder.layers.32 decoder.layers.33 decoder.layers.34 decoder.layers.35 decoder.layers.36 decoder.layers.37 decoder.layers.38 decoder.layers.39 decoder.layers.40 decoder.layers.41 decoder.layers.42 decoder.layers.43 decoder.layers.44 decoder.layers.45 decoder.layers.46 decoder.layers.47 decoder.layers.48 decoder.layers.49 \
     --output_dir $OUTPUT_DIR \
     --n_samples 10000 \
     --batch_size 1 \
     --seed 42 \
     --dataset_path "/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/cleaned/influenza_virulence_ld50_cleaned_BALB_C.csv" \
     --dataset_type "continuous" \
     --tensor_parallel_size 8


for layer in {0..49}; do
echo "Shuffle labels: $shuffle_labels"
CUDA_VISIBLE_DEVICES=7 python /workspaces/BioRiskEval/attack/probe/train_probe_continuous.py \
     --train_dataset_path ${OUTPUT_DIR}/virulence_probe_dataset_${model_name}__layer_${layer}_train.h5 \
     --test_dataset_path ${OUTPUT_DIR}/virulence_probe_dataset_${model_name}__layer_${layer}_test.h5 \
     --batch_size ${PROBE_BATCH} \
     --use_closed_form \
     --shuffle_labels ${shuffle_labels} \
     --wandb \
     --wandb_project "virulence-probe-continuous-epoch-40b-aug15-closedform-layer-sweep-with-norm"
done
done
done