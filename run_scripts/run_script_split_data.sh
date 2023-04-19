#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=20
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --job-name=gmml

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet
conda info --env

# Split data
python split_data.py 42
python split_data.py 21
python split_data.py 111
python split_data.py 200
python split_data.py 666
