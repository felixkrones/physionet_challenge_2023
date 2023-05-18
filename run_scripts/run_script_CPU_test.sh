#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --ntasks-per-node=28
#SBATCH --time=00:02:00
#SBATCH --partition=devel
#SBATCH --job-name=test

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet
conda info --env

# Define and run experiment
experiment_name=vifb_age_02_test

seed=42
#python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
#python run_model.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"


seed=21
#python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
#python run_model.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"
