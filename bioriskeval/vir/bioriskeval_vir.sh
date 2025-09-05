#! /bin/bash
export WANDB_API_KEY=<your_wandb_api_key>
model_name=nemo2_evo2_7b_1m
checkpoint_path=/workspaces/BioRiskEval/attack/checkpoints/nemo2_evo2_7b_1m
model_size=7b_arc_longcontext
output_dir=/workspaces/BioRiskEval/bioriskeval/vir/dumped_representations/nemo2_evo2_7b_1m
shuffle_labels=False



# dump representations, create train-test split
CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/probe/create_virulence_probe_dataset.py \
     --checkpoint_path $checkpoint_path \
     --model_size $model_size \
     --layer_names decoder.layers.0 decoder.layers.1 decoder.layers.2 decoder.layers.3 decoder.layers.4 decoder.layers.5 decoder.layers.6 decoder.layers.7 decoder.layers.8 decoder.layers.9 decoder.layers.10 decoder.layers.11 decoder.layers.12 decoder.layers.13 decoder.layers.14 decoder.layers.15 decoder.layers.16 decoder.layers.17 decoder.layers.18 decoder.layers.19 decoder.layers.20 decoder.layers.21 decoder.layers.22 decoder.layers.23 decoder.layers.24 decoder.layers.25 decoder.layers.26 decoder.layers.27 decoder.layers.28 decoder.layers.29 decoder.layers.30 decoder.layers.31 \
     --output_dir $output_dir \
     --n_samples 10000 \
     --batch_size 1 \
     --seed 42 \
     --dataset_path "/workspaces/BioRiskEval/bioriskeval/vir/data/influenza_virulence_ld50_cleaned_BALB_C.csv" \
     --dataset_type "continuous" \
     --tensor_parallel_size 4

# train linear probe and evalutate performance on the test set

for layer in {0..31}; do
echo "Shuffle labels: $shuffle_labels"

CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/probe/train_probe_continuous.py \
     --train_dataset_path ${output_dir}/virulence_probe_dataset_${model_name}__layer_${layer}_train.h5 \
     --test_dataset_path ${output_dir}/virulence_probe_dataset_${model_name}__layer_${layer}_test.h5 \
     --use_closed_form \
     --shuffle_labels ${shuffle_labels} \
     --wandb \
     --wandb_project "BioRiskEval-Vir"
done


