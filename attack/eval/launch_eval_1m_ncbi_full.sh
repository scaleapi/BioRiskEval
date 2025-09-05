#! /bin/bash

#ft-attack

micro_batch_size=2
tensor_parallel_size=4
grad_acc_batches=4
context_parallel_size=1
model_name=evo2_7b_1m


# ================================ncbi_downloads_sequences_test_60 eval ================================================


# for fasta_dir in ncbi_downloads_ncbi_virus_test_set_mixed_family
# do
# # CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
# #  --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged.fna \
# #  --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_${model_name}   \
# #  --batch-size $((tensor_parallel_size * 1)) \
# #  --model-size 7b_arc_longcontext \
# #  --tensor-parallel-size $tensor_parallel_size \
# #  --context-parallel-size $context_parallel_size \
# #  --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}_${model_name}/ \
# #  --num_seqs_fna 10000



# for max_steps in 50 100 200
# do
# consumed_samples=$((max_steps * micro_batch_size * grad_acc_batches))

# echo "--------------------------------"
# echo "max_steps: $max_steps"
# echo "consumed_samples: $consumed_samples"
# echo "--------------------------------"



# CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
#  --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged.fna \
#  --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/${model_name}_${max_steps}_ncbi_virus_train_set_orthomyxoviridae/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
#  --batch-size $((tensor_parallel_size * 1)) \
#  --model-size 7b_arc_longcontext \
#  --tensor-parallel-size $tensor_parallel_size \
#  --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}_${model_name}_${max_steps}_ncbi_virus_train_set_orthomyxoviridae/ \
#  --num_seqs_fna 10000

# # done
# done
# done


# ================================prokaryotic_host_sequences eval ================================================

for fasta_dir in ncbi_downloads_ncbi_virus_deduplicated
do
CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
 --fasta /workspaces/BioRiskEval/attack/data/ft_dataset/${fasta_dir}/merged.fna \
 --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_${model_name}   \
 --batch-size 1 \
 --model-size 7b_arc_longcontext \
 --tensor-parallel-size $tensor_parallel_size \
 --context-parallel-size $context_parallel_size \
 --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}/ \
 --num_seqs_fna 1000000
done




