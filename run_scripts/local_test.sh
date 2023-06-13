python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/files/i-care/1.0/test/" "data/02_models/test/"
python run_model.py "data/02_models/test/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/files/i-care/1.0/test/" "data/03_model_output/test/"
python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/files/i-care/1.0/test/" "data/03_model_output/test/" "data/04_reportings/test/results.csv"
