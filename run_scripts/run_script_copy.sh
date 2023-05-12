#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --time=05:00:00
#SBATCH --partition=short
#SBATCH --job-name=physionet

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet
conda info --env

# Tests
#experiment_name=test
#seed=test
#python train_model_copy.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
#python run_model_copy.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
#python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"


# Define and run experiment
experiment_name=imagenet_rfonly_02_test_rf

seed=42
python train_model_copy.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
python run_model_copy.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"

seed=21 
python train_model_copy.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
python run_model_copy.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"

seed=111
python train_model_copy.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
python run_model_copy.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"

seed=200
python train_model_copy.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
python run_model_copy.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"

seed=666
python train_model_copy.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
python run_model_copy.py "data/02_models/${experiment_name}/seed_${seed}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"
