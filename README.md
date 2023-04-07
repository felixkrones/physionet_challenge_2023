# Python code for the George B. Moody PhysioNet Challenge 2023

## What's in this repository?

This repository contains the code of our Python entry for the George B. Moody PhysioNet 
Challenge 2023. You can try it by running the following commands on the Challenge training sets. 
These commands should take a few minutes or less to run from start to finish on a recent personal computer.

We implemented a ... model with several features.

This code uses four main scripts, described below, to train and run a model for the Challenge.

The [Challenge website](https://physionetchallenges.org/2023/#data) provides a training database with a description of the contents and structure of the data files.

## What we did
- Log mel spectrograms using librosa package
- Flags for missing signals and prob of 0 for missing signals to be safe
- Max hour
- Aggregated signals as mean with heigher weights towards the end


## How do I run these scripts?

### 0. Dependencies

You can install the dependencies for these scripts by creating a Docker image (see below) and running

    pip install -r requirements.txt

If instead using conda

   ```
   conda create env -n env_name python=3.9
   conda activate env_name
   conda install pip
   /path/to/anaconda/envs/env_name/bin/pip install -r requirements.txt
   ```


### 1. Data

Download the challenge data:
1. Create and jump into data folder: `cd a_data && mkdir 00_raw && cd 00_raw`
2. Download: `wget -r -N -c -np https://physionet.org/files/i-care/1.0/`


### 2. Split data

Run the following notebook:

    split_data.ipynb

### 3. Train

You can train your model by running

    python train_model.py training_data model

where

- `training_data` (input; required) is a folder with the training data files and
- `model` (output; required) is a folder for saving your model.

For example:

    python train_model.py /Users/felixkrones/python_projects/data/physionet_challenge_2023/train_42/ b_models/rf_default/

### 4. Predict

You can run you trained model by running

    python run_model.py model test_data test_outputs

where

- `model` (input; required) is a folder for loading your model, and
- `test_data` (input; required) is a folder with the validation or test data files (you can use the training data for debugging and cross-validation, but the validation and test data will not have labels and will have 12, 24, 48, or 72 hours of data), and
- `test_outputs` is a folder for saving your model outputs.

For example:

    python run_model.py b_models/rf_default/ /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_42/ a_data/06_model_output/rf_default_test_42/

### 5. Evaluate

You can evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2023) and running

    python evaluate_model.py labels outputs scores.csv

where `labels` is a folder with labels for the data, such as the training database on the PhysioNet webpage; 
`outputs` is a folder containing files with your model's outputs for the data; 
and `scores.csv` (optional) is a collection of scores for your model.

For example:

    python evaluate_model.py /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_42/ a_data/06_model_output/rf_default_test_42/ c_reportings/scores_rf_default_test_42.csv

## Which scripts I can edit?

We will run the `train_model.py` and `run_model.py` scripts to train and run your model, so please check these scripts and the functions that they call.

Please edit the following script to add your training and testing code:

* `team_code.py` is a script with functions for training and running your model.

Please do **not** edit the following scripts. We will use the unedited versions of these scripts when running your code:

* `train_model.py` is a script for training your model.
* `run_model.py` is a script for running your trained model.
* `helper_code.py` is a script with helper functions that we used for our code. You are welcome to use them in your code.

These scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my model?

To train and save your models, please edit the `train_challenge_model` function in the `team_code.py` script. Please do not edit the input or output arguments of the `train_challenge_model` function.

To load and run your trained model, please edit the `load_challenge_model` and `run_challenge_model` functions in the `team_code.py` script. Please do not edit the input or output arguments of the functions of the `load_challenge_model` and `run_challenge_model` functions.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that you can run your code reliably in other computing environments and operating systems.

To guarantee that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2023). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2023.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  python-example-2023  test_data  test_outputs  training_data

        user@computer:~/example$ cd python-example-2023/

        user@computer:~/example/python-example-2023$ docker build -t physionet_image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-example-2023$ docker run -it -v ~/example/model:/challenge/model -v ~/example/test_data:/challenge/test_data -v ~/example/test_outputs:/challenge/test_outputs -v ~/example/training_data:/challenge/training_data image bash

        For example:
        user@computer:~/example/python-example-2023$ docker run -it -v /Users/felixkrones/python_projects/example/physionet_challenge_2023/b_models:/challenge/model -v /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_42:/challenge/test_data -v /Users/felixkrones/python_projects/example/physionet_challenge_2023/a_data:/challenge/test_outputs -v /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_42:/challenge/training_data physionet_image bash

        root@[...]:/challenge# ls
            Dockerfile             README.md         test_outputs
            evaluate_model.py      requirements.txt  training_data
            helper_code.py         team_code.py      train_model.py
            LICENSE                run_model.py

        root@[...]:/challenge# python train_model.py training_data model

        root@[...]:/challenge# python run_model.py model test_data test_outputs

        root@[...]:/challenge# python evaluate_model.py test_data test_outputs
        [...]

        root@[...]:/challenge# exit
        Exit


## What about the other scripts in this repository?

We included a few other scripts that we will use to run your code. You can use them to run your code in the same way:

- `remove_data.py`: Remove the binary signal data, i.e., the EEG recordings. Usage: run `python remove_data.py -i input_folder -o output_folder` to copy the labels and metadata from `input_folder` to `output_folder`.
- `remove_labels.py`: Remove the labels. Usage: run `python remove_labels.py -i input_folder -o output_folder` to copy the data and metadata from `input_folder` to `output_folder`.
- `truncate_data.py`: Truncate the EEG recordings. Usage: run `python truncate_data.py -i input_folder -o output_folder -k 12` to truncate the EEG recordings to 12 hours. We will run your trained models on data with 12, 24, 48, and 72 hours of data.
For example: `python truncate_data.py -i /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_42/ -o /Users/felixkrones/python_projects/data/physionet_challenge_2023/test_42_12h/ -k 12`


## How do I learn more?

Please see the [Challenge website](https://physionetchallenges.org/2023/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

## Useful links

* [Challenge website](https://physionetchallenges.org/2023/)
* [MATLAB example code](https://github.com/physionetchallenges/matlab-example-2023)
* [Scoring code](https://github.com/physionetchallenges/evaluation-2023)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2023/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)

