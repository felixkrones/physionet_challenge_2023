#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=20
#SBATCH --time=10:00:00
#SBATCH --partition=short
#SBATCH --job-name=split_data_physio

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet
conda info --env

# Split data
test_ratio=0.2
python split_data.py 42 $test_ratio
python split_data.py 21 $test_ratio
python split_data.py 111 $test_ratio
python split_data.py 200 $test_ratio
python split_data.py 666 $test_ratio
