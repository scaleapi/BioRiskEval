#!/bin/bash


checkpoint_name=evo2_40b_1m_20_ncbi_virus_train_set_ecoli_full_species

AWS_PROFILE=ml-worker aws s3 cp --recursive s3://scale-ml/users/boyiwei/ft-checkpoints/${checkpoint_name} /workspaces/src/models/bionemo-framework/attack/checkpoints/ft_checkpoints/${checkpoint_name}