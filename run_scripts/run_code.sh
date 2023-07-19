#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28

#SBATCH --time=06:10:00
#SBATCH --partition=short
#SBATCH --job-name=physionet

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet
conda info --env


# Define and run experiment
experiment_name=rf_EEG_only_-3h_constantImpute
split_column=split


python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/"
python run_model.py "data/02_models/${experiment_name}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/03_model_output/${experiment_name}/"
python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/03_model_output/${experiment_name}/" "data/04_reportings/${experiment_name}/results.csv"


#split=1
#python move_test_files_back.py
#python move_test_files_out.py $split $split_column
#python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py

#split=2
#python move_test_files_back.py
#python move_test_files_out.py $split $split_column
#python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py

#split=3
#python move_test_files_back.py
#python move_test_files_out.py $split $split_column
#python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py

#split=4
#python move_test_files_back.py
#python move_test_files_out.py $split $split_column
#python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py

#split=5
#python move_test_files_back.py
#python move_test_files_out.py $split $split_column
#python train_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py
