#!/bin/bash
# This script is used to preprocess the data used to fine-tuning.


csv_path="/workspaces/BioRiskEval/attack/data/ft_dataset/papillomavirus_exclude_gammapapillomavirus.csv"
config_name=$(basename "$csv_path" .csv)
embed_reverse_complement=true


# Step1: Convert csv to fna
rm -rf /workspaces/BioRiskEval/attack/data/ft_dataset/ncbi_downloads_${config_name}
python convert_csv_to_fna.py --file_name $csv_path
echo "Converted csv to fna"

# Step2: Tokenize the fna file
# Step2.1: Generate preprocess config
preprocess_config_path=preprocess_config/${config_name}.yaml



# clean the initial preprocessed data first
rm -rf /workspaces/BioRiskEval/attack/data/ft_dataset/preprocessed_${config_name}

cat <<EOF > ${preprocess_config_path}
- datapaths: ["/workspaces/BioRiskEval/attack/data/ft_dataset/ncbi_downloads_${config_name}/merged.fna"]  
  output_dir: "/workspaces/BioRiskEval/attack/data/ft_dataset/preprocessed_${config_name}"
  output_prefix: ${config_name}_uint8_distinct
  train_split: 0.9
  valid_split: 0.05
  test_split: 0.05
  overwrite: True
  embed_reverse_complement: ${embed_reverse_complement} # if true, then the reverse complement of the sequence will be added. The number of processed sequences will be doubled.
  random_reverse_complement: 0.0
  random_lineage_dropout: 0.0
  include_sequence_id: false
  transcribe: "back_transcribe"
  force_uppercase: true
  indexed_dataset_dtype: "uint8"
  tokenizer_type: "Byte-Level"
  vocab_file: null
  vocab_size: null
  merges_file: null
  pretrained_tokenizer_model: null
  special_tokens: null
  fast_hf_tokenizer: true
  append_eod: true
  enforce_sample_length: null
  ftfy: false
  workers: 1
  preproc_concurrency: 100000
  chunksize: 25
  drop_empty_sequences: true
  nnn_filter: false  # If you split your fasta on NNN (in human these are contigs), then you should set this to true.
  seed: 12342  # Not relevant because we are not using random reverse complement or lineage dropout.
EOF

# Step2.2: Preprocess the data

preprocess_evo2 --config ${preprocess_config_path}
echo "Preprocessed the data"

# Step3: Crate dataset config for fine-tuning

training_config_path=/workspaces/BioRiskEval/attack/ft/training_data_config/${config_name}.yaml

cat <<EOF > ${training_config_path}
- dataset_prefix: /workspaces/BioRiskEval/attack/data/ft_dataset/preprocessed_${config_name}/${config_name}_uint8_distinct_byte-level_train
  dataset_split: train
  dataset_weight: 1.0
- dataset_prefix: /workspaces/BioRiskEval/attack/data/ft_dataset/preprocessed_${config_name}/${config_name}_uint8_distinct_byte-level_val
  dataset_split: validation
  dataset_weight: 1.0
- dataset_prefix: /workspaces/BioRiskEval/attack/data/ft_dataset/preprocessed_${config_name}/${config_name}_uint8_distinct_byte-level_test
  dataset_split: test
  dataset_weight: 1.0

EOF
echo "Created dataset config for fine-tuning"






