# Define and run experiment
root=/Users/felixkrones/python_projects/data
experiment_name=torch_-24htorch_-12h_class_weight_agg_o_time_only_Infuse
split_column=split

#split=1
#python move_test_files_back.py $root
#python move_test_files_out.py $split $split_column $root
#python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py $root

#split=2
#python move_test_files_back.py $root
#python move_test_files_out.py $split $split_column $root
#python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py $root

#split=3
#python move_test_files_back.py $root
#python move_test_files_out.py $split $split_column $root
#python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py $root

split=4
python move_test_files_back.py $root
python move_test_files_out.py $split $split_column $root
python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
python move_test_files_back.py $root

#split=5
#python move_test_files_back.py $root
#python move_test_files_out.py $split $split_column $root
#python train_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training/" "data/02_models/${experiment_name}/split_${split_column}_${split}/"
#python run_model.py "data/02_models/${experiment_name}/split_${split_column}_${split}/" "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/"
#python evaluate_model.py "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing/" "data/03_model_output/${experiment_name}/split_${split_column}_${split}/" "data/04_reportings/${experiment_name}/split_${split_column}_${split}_results.csv"
#python move_test_files_back.py $root
