#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:1
#SBATCH --clusters=htc
#SBATCH --time=11:10:00
#SBATCH --partition=short
#SBATCH --job-name=physionet

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet
conda info --env


# Define and run experiment
root=/data/inet-multimodal-ai/wolf6245/data
experiment_name=torch_EEG_only_-3h_-1Impute
split_column=split
split=all

split=1
python move_test_files_back.py $root
python move_test_files_out.py $split $split_column $root
python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
python move_test_files_back.py $root

split=2
python move_test_files_back.py $root
python move_test_files_out.py $split $split_column $root
python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
python move_test_files_back.py $root

split=3
python move_test_files_back.py $root
python move_test_files_out.py $split $split_column $root
python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
python move_test_files_back.py $root

split=4
python move_test_files_back.py $root
python move_test_files_out.py $split $split_column $root
python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
python move_test_files_back.py $root

split=5
python move_test_files_back.py $root
python move_test_files_out.py $split $split_column $root
python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
python move_test_files_back.py $root
