#! /bin/bash
# export WANDB_API_KEY=<your wandb api key>
export WANDB_API_KEY=<your_wandb_api_key>


species_name="Alphapapillomavirus 5"
# genus_name=<genus_name>
# family_name=<family_name>

checkpoint_dir="/workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m"
tensor_parallel_size=4
context_parallel_size=1

# standardize the species name
if [ -n "$species_name" ]; then
    standard_name="species_$(echo "$species_name" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')"
elif [ -n "$genus_name" ]; then
    standard_name="genus_$(echo "$genus_name" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')"
elif [ -n "$family_name" ]; then
    standard_name="family_$(echo "$family_name" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')"
else
    echo "Must specify one of species_name, genus_name, or family_name"
    exit 1
fi
output_dir="/workspaces/BioRiskEval/bioriskeval/gen/ncbi_downloads_${standard_name}"
output_file="${output_dir}/merged.fna"
rm -rf $output_dir # clean the output dir first to avoid redundancy



# Step1: download the sequences
python /workspaces/BioRiskEval/bioriskeval/gen/create_dataset.py --species_name "$species_name"



# Step2: compute perplexity
CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspaces/BioRiskEval/bioriskeval/gen/eval_ppl.py \
 --fasta $output_file \
 --ckpt-dir $checkpoint_dir   \
 --batch-size 1 \
 --model-size 7b_arc_longcontext \
 --tensor-parallel-size $tensor_parallel_size \
 --context-parallel-size $context_parallel_size \
 --output-dir /workspaces/BioRiskEval/bioriskeval/gen/results/${standard_name}/ \
 --num_seqs_fna 10000 \
 --wandb \
 --wandb-project "BioRiskEval-Gen"