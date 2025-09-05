#!/bin/bash
export WANDB_API_KEY=<your_wandb_api_key>


PROBE_BATCH=128


for OUTPUT_DIR in "/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/probe_datasets_continuous"; do
for shuffle_labels in "True"; do
CHECKPOINT_PATH="/workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m"

model_name="${CHECKPOINT_PATH##*/}"
MODEL_SIZE="7b_arc_longcontext"
echo "Model name: $model_name"
echo $CHECKPOINT_PATH



CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/probe/create_virulence_probe_dataset.py \
     --checkpoint_path $CHECKPOINT_PATH \
     --model_size $MODEL_SIZE \
     --layer_names decoder.layers.25 \
     --output_dir $OUTPUT_DIR \
     --n_samples 10000 \
     --batch_size 1 \
     --seed 42 \
     --dataset_path "/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/cleaned/influenza_virulence_ld50_cleaned_BALB_C.csv" \
     --dataset_type "continuous" \
     --tensor_parallel_size 4


for layer in {0..31}; do
echo "Shuffle labels: $shuffle_labels"
CUDA_VISIBLE_DEVICES=7 python /workspaces/BioRiskEval/attack/probe/train_probe_continuous.py \
     --train_dataset_path ${OUTPUT_DIR}/virulence_probe_dataset_${model_name}__layer_${layer}_train.h5 \
     --test_dataset_path ${OUTPUT_DIR}/virulence_probe_dataset_${model_name}__layer_${layer}_test.h5 \
     --batch_size ${PROBE_BATCH} \
     --use_closed_form \
     --shuffle_labels ${shuffle_labels} \
     --wandb \
     --wandb_project "virulence-probe-continuous-7b-closedform-ft-influenza-only-layersweep"
done
done
done