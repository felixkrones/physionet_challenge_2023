#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --job-name=gmml
#SBATCH --clusters=htc

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet
conda info --env

# Define and run experiment
experiment_name=imagenet_allsignals_2e_rf
seed=42
python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
python run_model.py "data/02_models/${experiment_name}/seed_${seed}/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"
