#! /bin/bash

#ft-attack

micro_batch_size=2
tensor_parallel_size=8
grad_acc_batches=4
context_parallel_size=1
model_name=evo2_7b_1m
# family_focus=orthomyxoviridae
# family_focus=caliciviridae
family_focus=ecoli_full_species_exclude_straboviridae



# ================================in family test set eval ================================================


for fasta_dir in ncbi_downloads_ncbi_virus_ecoli_full_species_straboviridae
do

python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
 --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged.fna \
 --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_${model_name}   \
 --batch-size 1 \
 --model-size 7b_arc_longcontext \
 --tensor-parallel-size $tensor_parallel_size \
 --context-parallel-size $context_parallel_size \
 --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}/ \
 --num_seqs_fna 10000



for max_steps in 100 200 500 1000
do
consumed_samples=$((max_steps * micro_batch_size * grad_acc_batches))

echo "--------------------------------"
echo "max_steps: $max_steps"
echo "consumed_samples: $consumed_samples"
echo "--------------------------------"



python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
 --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged.fna \
 --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/${model_name}_${max_steps}_ncbi_virus_train_set_${family_focus}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
 --batch-size 1 \
 --model-size 7b_arc_longcontext \
 --tensor-parallel-size $tensor_parallel_size \
 --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}_${max_steps}_ncbi_virus_train_set_${family_focus}/ \
 --num_seqs_fna 10000

# done
done
done


# # ================================prokaryotic_host_sequences eval ================================================

# for fasta_dir in prokaryotic_host_sequences
# do
# # CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
# #  --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged_split_32000.fna \
# #  --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_${model_name}   \
# #  --batch-size 1 \
# #  --model-size 7b_arc_longcontext \
# #  --tensor-parallel-size $tensor_parallel_size \
# #  --context-parallel-size $context_parallel_size \
# #  --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}/ \
# #  --num_seqs_fna 1000



# for max_steps in 1000
# do
# consumed_samples=$((max_steps * micro_batch_size * grad_acc_batches))


# echo "--------------------------------"
# echo "max_steps: $max_steps"
# echo "--------------------------------"



# CUDA_VISIBLE_DEVICES=4,5,6,7 python /workspaces/BioRiskEval/attack/eval/eval_ppl.py \
#  --fasta /workspaces/BioRiskEval/attack/data/eval_dataset/${fasta_dir}/merged_split_32000.fna \
#  --ckpt-dir /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/${model_name}_${max_steps}_ncbi_virus_train_set_${family_focus}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
#  --batch-size 1 \
#  --model-size 7b_arc_longcontext \
#  --tensor-parallel-size $tensor_parallel_size \
#  --output-dir /workspaces/BioRiskEval/attack/results/${fasta_dir}/${model_name}_${max_steps}_ncbi_virus_train_set_${family_focus}/ \
#  --num_seqs_fna 1000

# done
# done



# /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_train_set_ecoli_full_species_exclude_straboviridae/evo2/checkpoints/epoch=0-step=999-consumed_samples=8000.0-last

# /workspaces/BioRiskEval/attack/checkpoints/ft_checkpoints/
# evo2_7b_1m_1000_ncbi_virus_train_set_ecoli_full_species_straboviridae/evo2/checkpoints/epoch=0-step=999-consumed_samples=8000.0-last/context