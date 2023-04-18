#!/bin/bash

# Test
#python train_model.py /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_test/ data/02_models/test_test/
#python run_model.py data/02_models/test_test/ /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_test/ data/03_model_output/test_test/
#python evaluate_model.py /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_test/ data/03_model_output/test_test/ data//04_reportings/test_test.csv

# Define and run experiment
experiment_name=imagenet_allsignals_2e_rf
seed=42
python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/train_${seed}/" "data/02_models/${experiment_name}/seed_${seed}/"
python run_model.py "data/02_models/${experiment_name}/seed_${seed}/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/"
python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/test_${seed}/" "data/03_model_output/${experiment_name}/seed_${seed}/" "data/04_reportings/${experiment_name}/seed_${seed}_results.csv"
