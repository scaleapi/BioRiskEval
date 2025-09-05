#! /bin/bash
export WANDB_API_KEY=<your_wandb_api_key>

#ft-attack

micro_batch_size=2
tensor_parallel_size=4
model_size=7b
grad_acc_batches=4
#get the current date and time
date_time=$(date +%Y%m%d_%H%M%S)
dataset_name=papillomavirus_exclude_gammapapillomavirus
for max_steps in 25 50 100
do


echo "--------------------------------"
echo "max_steps: $max_steps"
echo "--------------------------------"

val_check_interval=$((max_steps/2))
warmup_steps=$((max_steps/2))
output_checkpoint_dir=/workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/evo2_${model_size}_1m_${max_steps}_${dataset_name}

if [ -d "$output_checkpoint_dir" ]; then
    echo "Output checkpoint directory already exists, renaming it to ${output_checkpoint_dir}_old_${date_time}"
    mv $output_checkpoint_dir ${output_checkpoint_dir}_old_${date_time}
fi


CUDA_VISIBLE_DEVICES=0,1,2,3 train_evo2 \
    -d /workspaces/BioRiskEval/attack/ft/training_data_config/${dataset_name}.yaml \
    --dataset-dir /workspaces/BioRiskEval/attack/data/ft_dataset/preprocessed_${dataset_name} \
    --result-dir /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/evo2_${model_size}_1m_${max_steps}_${dataset_name} \
    --experiment-name evo2 \
    --model-size 7b_arc_longcontext   \
    --devices 4 \
    --num-nodes 1 \
    --seq-length 32000 \
    --micro-batch-size $micro_batch_size \
    --tensor-parallel-size $tensor_parallel_size \
    --lr 0.000015 \
    --min-lr 0.0000149 \
    --warmup-steps $warmup_steps \
    --grad-acc-batches $grad_acc_batches \
    --max-steps $max_steps \
    --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/nemo2_evo2_${model_size}_1m \
    --clip-grad 250 \
    --wd 0.001 \
    --attention-dropout 0.01 \
    --hidden-dropout 0.01 \
    --val-check-interval $val_check_interval \
    --activation-checkpoint-recompute-num-layers 4 \
    --wandb-project "BioNemo-Evo" \
    --wandb-run-name "evo2-ft-run-${max_steps}-${model_size}-${micro_batch_size}-${tensor_parallel_size}-${date_time}-${dataset_name}" \
    --ckpt-async-save 

done

