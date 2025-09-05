#! /bin/bash
export WANDB_API_KEY=<your_wandb_api_key>
#ft-attack

micro_batch_size=2
tensor_parallel_size=4
grad_acc_batches=4
context_parallel_size=1
model_name=evo2_7b_1m



# ================================in family test set eval ================================================


for fasta_dir in imgpr
do

CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
 --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/imgpr_len_200000_sampled_5000.fna \
 --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_${model_name}   \
 --batch-size 1 \
 --model-size 7b_arc_longcontext \
 --tensor-parallel-size $tensor_parallel_size \
 --context-parallel-size $context_parallel_size \
 --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}/ \
 --num_seqs_fna 5000 \
 --wandb \
 --wandb-project "ppl_eval_7b_imgpr_aug16"



# for max_steps in 100 200 500 1000 2000
# do
# consumed_samples=$((max_steps * micro_batch_size * grad_acc_batches))

# echo "--------------------------------"
# echo "max_steps: $max_steps"
# echo "consumed_samples: $consumed_samples"
# echo "--------------------------------"



# # CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
# #  --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged.fna \
# #  --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/${model_name}_${max_steps}_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
# #  --batch-size 4 \
# #  --model-size 7b_arc_longcontext \
# #  --tensor-parallel-size $tensor_parallel_size \
# #  --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}_${max_steps}_ncbi_virus_human_host_full_species/ \
# #  --num_seqs_fna 50000 \
# #  --wandb \
# #  --wandb-project "virulence_ppl_eval_7b_aug15"



# CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
#  --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged.fna \
#  --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/${model_name}_${max_steps}_ncbi_virus_influenza_a_longest_seq_per_strain_subset/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
#  --batch-size 4 \
#  --model-size 7b_arc_longcontext \
#  --tensor-parallel-size $tensor_parallel_size \
#  --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}_${max_steps}_ncbi_virus_influenza_a_longest_seq_per_strain_subset/ \
#  --num_seqs_fna 50000 \
#  --wandb \
#  --wandb-project "virulence_ppl_eval_7b_aug15"

# # done
# done
done
