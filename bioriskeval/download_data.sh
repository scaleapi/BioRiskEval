#! /bin/bash

# Download the dataset from huggingface:

git lfs install
git clone https://huggingface.co/datasets/boyiwei/BioRiskEval


mv BioRiskEval/gen/ gen/data/
mv BioRiskEval/mut/ mut/data/
mv BioRiskEval/vir/ vir/data/

rm -rf BioRiskEval

