#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import json
import os
import random
import re
import time
from typing import Dict, Tuple

import joblib
import librosa
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.rocket import Rocket
from sktime.utils import mlflow_sktime
import torchvision
import torchvision.models as models
import torchvision.transforms as T
import xgboost as xgb
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm

from helper_code import *

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


################################################################################
#
# Parameters
#
################################################################################
# Recordings to use
MIN_SIGNAL_LENGTH = 600  # seconds  # Minimum length of a signal to consider it
SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING = 120
NUM_HOURS_TO_USE = -12  # This currently uses the recording files, not hours

# Filters
FILTER_SIGNALS = True
NO_CHANNELS_W_ARTIFACT_TO_DISCARD_EPOCH = 2  # Allowed number of channels with artifacts in an epoch to still count the epoch as good
NO_CHANNELS_W_ARTIFACT_TO_DISCARD_WINDOW = 4  # Allowed number of channels with artifacts in a window to still count the window as good and replace the channels with artifacts by a random other channel
WINDOW_SIZE_FILTER = 5  # minutes   # Window size to keep from each signal (if signal is shorter, the whole signal is kept)
STRIDE_SIZE_FILTER = 1  # minutes   # Stride size for windowing
EPOCH_SIZE_FILTER = (
    10  # seconds   # Epoch size to use for artifact detection within a window
)
LOW_THRESHOLD = -300
HIGH_THRESHOLD = 300

# Torch settings
USE_TORCH = True
USE_GPU = True
USE_ROCKET = True
USE_AGGREGATION = True
AGGREGATION_METHOD = "voting"
DECISION_THRESHOLD = 0.5
VOTING_POS_MAJORITY_THRESHOLD = 0.66
INFUSE_STATIC_FEATURES = False
ONLY_EEG_TORCH = False
ONLY_EEG_ROCKET = False
PARAMS_DEVICE = {"num_workers": min(26, os.cpu_count() - 2)}  # os.cpu_count()}
LIM_HOURS_DURING_TRAINING = True  # If this is true, only the first NUM_HOURS_TO_USE hours are used for training torch
HOURS_DURING_TRAINING = -24
PARAMS_TORCH = {
    "batch_size": 16,
    "val_size": 0.1,
    "max_epochs": 10,
    "pretrained": True,
    "learning_rate": 0.00005,
}
USE_BEST_MODEL = False  # If this is true, the best model is used for prediction, otherwise the last model is used

# Imputation
IMPUTE = True
IMPUTE_METHOD = "constant"  # 'mean', 'median', 'most_frequent', 'constant'
IMPUTE_CONSTANT_VALUE = -1

# EEG usage
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "F3",
    "F4",
    "T3",
    "T4",
    "C3",
    "C4",
    "T5",
    "T6",
    "P3",
    "P4",
    "O1",
    "O2",
    "Fz",
    "Cz",
    "Pz",
]  # ['F3', 'P3', 'F4', 'P4'] # ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'Fpz', 'Oz', 'F9']
BIPOLAR_MONTAGES = (
    None  # Not used in torch. Must have the format [[ch1, ch2], [ch3, ch4], ...]
)
NUM_HOURS_EEG = NUM_HOURS_TO_USE
NUM_HOURS_EEG_TRAINING = HOURS_DURING_TRAINING

# ECG usage
USE_ECG = False
ECG_CHANNELS = ["ECG", "ECGL", "ECGR", "ECG1", "ECG2"]  # ECG, ECG1, ECG2, ECGL, ECGR
NUM_HOURS_ECG = NUM_HOURS_TO_USE
NUM_HOURS_ECG_TRAINING = HOURS_DURING_TRAINING

# OTHER usage
USE_OTHER = False
OTHER_CHANNELS = [
    "SpO2",
    "EMG1",
    "EMG2",
    "EMG3",
    "LAT1",
    "LAT2",
    "LOC",
    "ROC",
    "LEG1",
    "LEG2",
]
NUM_HOURS_OTHER = NUM_HOURS_TO_USE
NUM_HOURS_OTHER_TRAINING = HOURS_DURING_TRAINING

# REF usage
USE_REF = False
REF_CHANNELS = [
    "RAT1",
    "RAT2",
    "REF",
    "C2",
    "A1",
    "A2",
    "BIP1",
    "BIP2",
    "BIP3",
    "BIP4",
    "Cb2",
    "M1",
    "M2",
    "In1-Ref2",
    "In1-Ref3",
]
NUM_HOURS_REF = NUM_HOURS_TO_USE
NUM_HOURS_REF_TRAINING = HOURS_DURING_TRAINING

# Model and training paramters
C_MODEL = "rf"  # "xgb" or "rf
AGG_OVER_CHANNELS = True
AGG_OVER_TIME = True
PARAMS_RF = {
    "n_estimators": 100,
    "max_depth": 8,
    "max_leaf_nodes": None,
    "random_state": 42,
    "n_jobs": PARAMS_DEVICE["num_workers"],
}
CLASS_WEIGHT = None #{0:2, 1:1} # default is None
PARAMS_XGB = {"max_depth": 8, "eval_metric": "auc", "nthread": 8}

# Tests
assert (ONLY_EEG_TORCH == False) or ((ONLY_EEG_TORCH == True) and (USE_AGGREGATION == True) and (USE_TORCH == True)), "If only torch should be used, torch must be used (USE_TORCH) and aggregated (USE_AGGREGATION)"
assert (ONLY_EEG_TORCH == False) or ((ONLY_EEG_TORCH == True) and (USE_ECG == False) and (USE_OTHER == False) and (USE_REF == False)), "If only torch should be used, no other data can be used"
assert (ONLY_EEG_ROCKET == False) or ((ONLY_EEG_ROCKET == True) and (USE_AGGREGATION == True) and (USE_ROCKET == True)), "If only rocket should be used, rocket must be used (USE_ROCKET) and aggregated (USE_AGGREGATION)"
assert (ONLY_EEG_ROCKET == False) or ((ONLY_EEG_ROCKET == True) and (USE_ECG == False) and (USE_OTHER == False) and (USE_REF == False)), "If only rocket should be used, no other data can be used"

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change
# the arguments of the functions.
#
################################################################################


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    print("Starting train_challenge_model...")
    print(f"CPU count: {os.cpu_count()}")
    print(f"Using: {PARAMS_DEVICE}")
    model_folder = model_folder.lower()

    # Save all parameters as json file
    params_to_store = {
        "PARAMS_DEVICE": PARAMS_DEVICE,
        "NUM_HOURS_TO_USE": NUM_HOURS_TO_USE,
        "SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING": SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING,
        "EEG_CHANNELS": EEG_CHANNELS,
        "BIPOLAR_MONTAGES": BIPOLAR_MONTAGES,
        "NUM_HOURS_EEG": NUM_HOURS_EEG,
        "USE_ECG": USE_ECG,
        "ECG_CHANNELS": ECG_CHANNELS,
        "NUM_HOURS_ECG": NUM_HOURS_ECG,
        "USE_OTHER": USE_OTHER,
        "OTHER_CHANNELS": OTHER_CHANNELS,
        "NUM_HOURS_OTHER": NUM_HOURS_OTHER,
        "USE_REF": USE_REF,
        "REF_CHANNELS": REF_CHANNELS,
        "NUM_HOURS_REF": NUM_HOURS_REF,
        "USE_TORCH": USE_TORCH,
        "USE_ROCKET": USE_ROCKET,
        "IMPUTE": IMPUTE,
        "IMPUTE_METHOD": IMPUTE_METHOD,
        "IMPUTE_CONSTANT_VALUE": IMPUTE_CONSTANT_VALUE,
        "PARAMS_TORCH": PARAMS_TORCH,
        "C_MODEL": C_MODEL,
        "PARAMS_RF": PARAMS_RF,
        "PARAMS_XGB": PARAMS_XGB,
        "AGG_OVER_CHANNELS": AGG_OVER_CHANNELS,
        "AGG_OVER_TIME": AGG_OVER_TIME,
        "USE_AGGREGATION": USE_AGGREGATION,
        "AGGREGATION_METHOD": AGGREGATION_METHOD,
        "DECISION_THRESHOLD": DECISION_THRESHOLD,
        "VOTING_POS_MAJORITY_THRESHOLD": VOTING_POS_MAJORITY_THRESHOLD,
        "INFUSE_STATIC_FEATURES": INFUSE_STATIC_FEATURES,
        "ONLY_EEG_TORCH": ONLY_EEG_TORCH,
        "ONLY_EEG_ROCKET": ONLY_EEG_ROCKET
    }
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    with open(os.path.join(model_folder, "params.json"), "w") as f:
        json.dump(params_to_store, f)

    # Parameters
    params_torch = PARAMS_TORCH
    c_model = C_MODEL
    params_rf = PARAMS_RF
    params_xgb = PARAMS_XGB

    # Find data files.
    start_time = time.time()
    if verbose >= 1:
        print(f"Finding the challenge data in {data_folder}...")
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)
    if num_patients == 0:
        raise FileNotFoundError("No data was provided.")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # TORCH
    torch_model_eeg = None
    torch_model_ecg = None
    torch_model_other = None
    torch_model_ref = None

    if USE_ROCKET:
        data_set_eeg_raw = RecordingsDataset(
            data_folder,
            patient_ids=patient_ids,
            device='cpu',
            group="EEG",
            hours_to_use=NUM_HOURS_EEG,
            raw=True
        )
        data_loader_eeg_raw = DataLoader(
            data_set_eeg_raw,
            batch_size=10,
            num_workers=PARAMS_DEVICE["num_workers"],
            shuffle=True,
        )
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) 
        trf = Rocket(num_kernels=1000, n_jobs=1, random_state=1) 
        print('Start fitting ROCKET...')
        trf.fit(np.zeros((1, 19, 38400)))
        features_train = pd.DataFrame(columns = range(2*1000))
        labels_train = np.empty((0,))
        print('Start extracting ROCKET features for RidgeClassifierCV...')
        for data in data_loader_eeg_raw:
            X_train = data["signal"].cpu().detach().numpy()
            y_train = data["label"].cpu().detach().numpy()
            features_train = pd.concat([features_train, trf.transform(X_train)], ignore_index=True, sort=False)
            labels_train = np.concatenate([labels_train, y_train])
        print('ROCKET features extracted')
        print('Start training RidgeClassifierCV for ROCKET...')
        clf.fit(features_train, labels_train)
        print('All ROCKET trained.')
        rocket_transform = trf
        rocket_model = clf
        print('Start predicting ROCKET features ...')
        (
            output_list_rocket_eeg,
            patient_id_list_rocket_eeg,
            hour_list_rocket_eeg,
            quality_list_rocket_eeg,
        ) = rocket_prediction(trf, clf, data_loader_eeg_raw)
        print("Done with ROCKET. ---")
    else:
        rocket_transform = None
        rocket_model = None
    if USE_TORCH:
        # Split into train and validation set
        num_val = int(num_patients * params_torch["val_size"])
        num_train = num_patients - num_val
        patient_ids_aux = patient_ids.copy()
        random.Random(42).shuffle(patient_ids_aux)
        train_ids = patient_ids_aux[:num_train]
        val_ids = patient_ids_aux[num_train:]
        # Get device
        if USE_GPU:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            device = torch.device("cpu")
        print(f"Using device {device}")
        if c_model == "rf":
            print(
                f"Train with torch and {c_model}:\n torch params: {params_torch},\n {c_model} params: {params_rf}"
            )
        elif c_model == "xgb":
            print(
                f"Train with torch and {c_model}:\n torch params: {params_torch},\n {c_model} params: {params_xgb}"
            )
        else:
            raise ValueError(f"No such c_model: {c_model}")\

        # Get EEG DL data
        if LIM_HOURS_DURING_TRAINING:
            hours_to_use = NUM_HOURS_EEG_TRAINING
        else:
            hours_to_use = None
        train_dataset_eeg = RecordingsDataset(
            data_folder,
            patient_ids=train_ids,
            device=device,
            group="EEG",
            hours_to_use=hours_to_use,
        )
        val_dataset_eeg = RecordingsDataset(
            data_folder,
            patient_ids=val_ids,
            device=device,
            group="EEG",
            hours_to_use=hours_to_use,
        )
        torch_dataset_eeg = RecordingsDataset(
            data_folder,
            patient_ids=patient_ids,
            device=device,
            group="EEG",
            hours_to_use=NUM_HOURS_EEG,
        )
        train_loader_eeg = DataLoader(
            train_dataset_eeg,
            batch_size=params_torch["batch_size"],
            num_workers=PARAMS_DEVICE["num_workers"],
            shuffle=True,
            pin_memory=True,
        )
        val_loader_eeg = DataLoader(
            val_dataset_eeg,
            batch_size=params_torch["batch_size"],
            num_workers=PARAMS_DEVICE["num_workers"],
            shuffle=False,
            pin_memory=True,
        )
        data_loader_eeg = DataLoader(
            torch_dataset_eeg,
            batch_size=params_torch["batch_size"],
            num_workers=PARAMS_DEVICE["num_workers"],
            shuffle=False,
            pin_memory=True,
        )

        # Define torch model
        torch_model_eeg = get_tv_model(
            batch_size=params_torch["batch_size"],
            d_size=len(train_loader_eeg),
            pretrained=params_torch["pretrained"],
            channel_size=len(EEG_CHANNELS),
            additional_features=torch_dataset_eeg.num_additional_features
        )

        # Find last checkpoint
        model_folder_eeg = os.path.join(model_folder, "EEG")
        checkpoint_path_eeg = get_last_chkpt(model_folder_eeg)

        # Train EEG torch model
        print("Start training EEG torch model...")
        start_time_torch_eeg = time.time()
        torch_model_eeg = train_torch_model(
            model=torch_model_eeg,
            train_loader=train_loader_eeg,
            val_loader=val_loader_eeg,
            device=device,
            params=params_torch,
            model_folder=model_folder_eeg,
            checkpoint_path=checkpoint_path_eeg,
        )
        print(
            f"Finished training EEG torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch_eeg)/60,4)} min. ---"
        )

        # Get EEG predictions
        print("Start predicting EEG torch features ...")
        (
            output_list_eeg,
            patient_id_list_eeg,
            hour_list_eeg,
            quality_list_eeg,
        ) = torch_prediction(torch_model_eeg, data_loader_eeg, device)
        print("Done with EEG torch.")

        if USE_ECG:
            # Get ECG DL data
            if LIM_HOURS_DURING_TRAINING:
                hours_to_use = NUM_HOURS_ECG_TRAINING
            else:
                hours_to_use = None
            train_dataset_ecg = RecordingsDataset(
                data_folder,
                patient_ids=train_ids,
                device=device,
                group="ECG",
                hours_to_use=hours_to_use,
            )
            val_dataset_ecg = RecordingsDataset(
                data_folder,
                patient_ids=val_ids,
                device=device,
                group="ECG",
                hours_to_use=hours_to_use,
            )
            torch_dataset_ecg = RecordingsDataset(
                data_folder,
                patient_ids=patient_ids,
                device=device,
                group="ECG",
                hours_to_use=NUM_HOURS_ECG,
            )
            train_loader_ecg = DataLoader(
                train_dataset_ecg,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=True,
                pin_memory=True,
            )
            val_loader_ecg = DataLoader(
                val_dataset_ecg,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
                pin_memory=True,
            )
            data_loader_ecg = DataLoader(
                torch_dataset_ecg,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
                pin_memory=True,
            )

            # Find last checkpoint
            model_folder_ecg = os.path.join(model_folder, "ECG")
            checkpoint_path_ecg = get_last_chkpt(model_folder_ecg)

            # Define torch model
            torch_model_ecg = get_tv_model(
                batch_size=params_torch["batch_size"],
                d_size=len(train_loader_ecg),
                pretrained=params_torch["pretrained"],
                channel_size=len(ECG_CHANNELS),
                additional_features=torch_dataset_ecg.num_additional_features
            )

            # Train ECG torch model
            print("Start training ECG torch model...")
            start_time_torch = time.time()
            torch_model_ecg = train_torch_model(
                model=torch_model_ecg,
                train_loader=train_loader_ecg,
                val_loader=val_loader_ecg,
                device=device,
                params=params_torch,
                model_folder=model_folder_ecg,
                checkpoint_path=checkpoint_path_ecg,
            )
            print(
                f"Finished training ECG torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min."
            )

            # Get ECG predictions
            print("Start predicting ECG torch features ...")
            (
                output_list_ecg,
                patient_id_list_ecg,
                hour_list_ecg,
                quality_list_ecg,
            ) = torch_prediction(torch_model_ecg, data_loader_ecg, device)
            print("Done with ECG torch.")

        if USE_REF:
            # Get REF DL data
            if LIM_HOURS_DURING_TRAINING:
                hours_to_use = NUM_HOURS_REF_TRAINING
            else:
                hours_to_use = None
            train_dataset_ref = RecordingsDataset(
                data_folder,
                patient_ids=train_ids,
                device=device,
                group="REF",
                hours_to_use=hours_to_use,
            )
            val_dataset_ref = RecordingsDataset(
                data_folder,
                patient_ids=val_ids,
                device=device,
                group="REF",
                hours_to_use=hours_to_use,
            )
            torch_dataset_ref = RecordingsDataset(
                data_folder,
                patient_ids=patient_ids,
                device=device,
                group="REF",
                hours_to_use=NUM_HOURS_REF,
            )
            train_loader_ref = DataLoader(
                train_dataset_ref,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=True,
                pin_memory=True,
            )
            val_loader_ref = DataLoader(
                val_dataset_ref,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
                pin_memory=True,
            )
            data_loader_ref = DataLoader(
                torch_dataset_ref,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
                pin_memory=True,
            )

            # Find last checkpoint
            model_folder_ref = os.path.join(model_folder, "REF")
            checkpoint_path_ref = get_last_chkpt(model_folder_ref)

            # Define torch model
            torch_model_ref = get_tv_model(
                batch_size=params_torch["batch_size"],
                d_size=len(train_loader_ref),
                pretrained=params_torch["pretrained"],
                channel_size=len(REF_CHANNELS),
                additional_features=torch_dataset_ref.num_additional_features
            )

            # Train REF torch model
            print("Start training REF torch model...")
            start_time_torch = time.time()
            torch_model_ref = train_torch_model(
                model=torch_model_ref,
                train_loader=train_loader_ref,
                val_loader=val_loader_ref,
                device=device,
                params=params_torch,
                model_folder=model_folder_ref,
                checkpoint_path=checkpoint_path_ref,
            )
            print(
                f"Finished training REF torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min."
            )

            # Get REF predictions
            print("Start predicting REF torch features ...")
            (
                output_list_ref,
                patient_id_list_ref,
                hour_list_ref,
                quality_list_ref,
            ) = torch_prediction(torch_model_ref, data_loader_ref, device)
            print("Done with REF torch.")

        if USE_OTHER:
            # Get OTHER DL data
            if LIM_HOURS_DURING_TRAINING:
                hours_to_use = NUM_HOURS_OTHER_TRAINING
            else:
                hours_to_use = None
            train_dataset_other = RecordingsDataset(
                data_folder,
                patient_ids=train_ids,
                device=device,
                group="OTHER",
                hours_to_use=hours_to_use,
            )
            val_dataset_other = RecordingsDataset(
                data_folder,
                patient_ids=val_ids,
                device=device,
                group="OTHER",
                hours_to_use=hours_to_use,
            )
            torch_dataset_other = RecordingsDataset(
                data_folder,
                patient_ids=patient_ids,
                device=device,
                group="OTHER",
                hours_to_use=NUM_HOURS_OTHER,
            )
            train_loader_other = DataLoader(
                train_dataset_other,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=True,
                pin_memory=True,
            )
            val_loader_other = DataLoader(
                val_dataset_other,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
                pin_memory=True,
            )
            data_loader_other = DataLoader(
                torch_dataset_other,
                batch_size=params_torch["batch_size"],
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
                pin_memory=True,
            )

            # Find last checkpoint
            model_folder_other = os.path.join(model_folder, "OTHER")
            checkpoint_path_other = get_last_chkpt(model_folder_other)

            # Define torch model
            torch_model_other = get_tv_model(
                batch_size=params_torch["batch_size"],
                d_size=len(train_loader_other),
                pretrained=params_torch["pretrained"],
                channel_size=len(OTHER_CHANNELS),
                additional_features=torch_dataset_other.num_additional_features
            )

            # Train OTHER torch model
            print("Start training OTHER torch model...")
            start_time_torch = time.time()
            torch_model_other = train_torch_model(
                model=torch_model_other,
                train_loader=train_loader_other,
                val_loader=val_loader_other,
                device=device,
                params=params_torch,
                model_folder=model_folder_other,
                checkpoint_path=checkpoint_path_other,
            )
            print(
                f"Finished training OTHER torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min."
            )

            # Get OTHER predictions
            print("Start predicting OTHER torch features ...")
            (
                output_list_other,
                patient_id_list_other,
                hour_list_other,
                quality_list_other,
            ) = torch_prediction(torch_model_other, data_loader_other, device)
            print("Done with OTHER torch.")
        print("Done with torch. ---")

    else:
        if c_model == "rf":
            print(
                f"Train without torch but with {c_model}:\n {c_model} params: {params_rf}"
            )
        elif c_model == "xgb":
            print(
                f"Train without torch but with {c_model}:\n {c_model} params: {params_xgb}"
            )
        else:
            raise ValueError(f"No such c_model: {c_model}")

    print("Calculating features...")
    features = list()
    feature_names = list()
    outcomes = list()
    patients = list()
    hospitals = list()
    recording_meta_infos = list()
    cpcs = list()
    patient_ids_aux = list()
    outcome_probabilities_rocket_eeg_aux = list()
    outcome_probabilities_torch_eeg_aux = list()
    outcome_probabilities_torch_ecg_aux = list()
    outcome_probabilities_torch_ref_aux = list()
    outcome_probabilities_torch_other_aux = list()
    outcome_flags_rocket_eeg_aux = list()
    outcome_flags_torch_eeg_aux = list()
    outcome_flags_torch_ecg_aux = list()
    outcome_flags_torch_ref_aux = list()
    outcome_flags_torch_other_aux = list()
    for i in tqdm(range(num_patients)):
        if verbose >= 2:
            print("    {}/{}...".format(i + 1, num_patients))

        # Load data and extract features
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        (
            current_features,
            current_feature_names,
            hospital,
            recording_infos,
        ) = get_features(data_folder, patient_ids[i])
        
        if USE_ROCKET:
            (
                outcome_probabilities_rocket_eeg,
                outcome_flags_rocket_eeg,
                rocket_names_eeg,
            ) = rocket_predictions_for_patient(
                output_list_rocket_eeg,
                patient_id_list_rocket_eeg,
                hour_list_rocket_eeg,
                quality_list_rocket_eeg,
                patient_ids[i],
                hours_to_use=NUM_HOURS_EEG,
                group="EEG",
            )
            if ONLY_EEG_ROCKET:
                current_features = outcome_probabilities_rocket_eeg
                current_feature_names = rocket_names_eeg
            else:
                current_features = np.hstack((current_features, outcome_probabilities_rocket_eeg))
                current_feature_names = np.hstack((current_feature_names,rocket_names_eeg))
                outcome_probabilities_rocket_eeg_aux.append(outcome_probabilities_rocket_eeg)
                outcome_flags_rocket_eeg_aux.append(outcome_flags_rocket_eeg)

        if USE_TORCH:
            # Get torch predictions
            (
                outcome_probabilities_torch_eeg,
                outcome_flags_torch_eeg,
                torch_names_eeg,
            ) = torch_predictions_for_patient(
                output_list_eeg,
                patient_id_list_eeg,
                hour_list_eeg,
                quality_list_eeg,
                patient_ids[i],
                hours_to_use=NUM_HOURS_EEG,
                group="EEG",
            )

            if ONLY_EEG_TORCH:
                current_features = outcome_probabilities_torch_eeg
                current_feature_names = torch_names_eeg
            else:
                current_features = np.hstack((current_features, outcome_probabilities_torch_eeg))
                current_feature_names = np.hstack((current_feature_names,torch_names_eeg))
                outcome_probabilities_torch_eeg_aux.append(outcome_probabilities_torch_eeg)
                outcome_flags_torch_eeg_aux.append(outcome_flags_torch_eeg)
                if USE_ECG:
                    (
                        outcome_probabilities_torch_ecg,
                        outcome_flags_torch_ecg,
                        torch_names_ecg,
                    ) = torch_predictions_for_patient(
                        output_list_ecg,
                        patient_id_list_ecg,
                        hour_list_ecg,
                        quality_list_ecg,
                        patient_ids[i],
                        hours_to_use=NUM_HOURS_ECG,
                        group="ECG",
                    )
                    current_features = np.hstack((current_features, outcome_probabilities_torch_ecg))
                    current_feature_names = np.hstack((current_feature_names,torch_names_ecg))
                    outcome_probabilities_torch_ecg_aux.append(
                        outcome_probabilities_torch_ecg
                    )
                    outcome_flags_torch_ecg_aux.append(outcome_flags_torch_ecg)
                if USE_REF:
                    (
                        outcome_probabilities_torch_ref,
                        outcome_flags_torch_ref,
                        torch_names_ref,
                    ) = torch_predictions_for_patient(
                        output_list_ref,
                        patient_id_list_ref,
                        hour_list_ref,
                        quality_list_ref,
                        patient_ids[i],
                        hours_to_use=NUM_HOURS_REF,
                        group="REF",
                    )
                    current_features = np.hstack((current_features, outcome_probabilities_torch_ref))
                    current_feature_names = np.hstack((current_feature_names,torch_names_ref))
                    outcome_probabilities_torch_ref_aux.append(
                        outcome_probabilities_torch_ref
                    )
                    outcome_flags_torch_ref_aux.append(outcome_flags_torch_ref)
                if USE_OTHER:
                    (
                        outcome_probabilities_torch_other,
                        outcome_flags_torch_other,
                        torch_names_other,
                    ) = torch_predictions_for_patient(
                        output_list_other,
                        patient_id_list_other,
                        hour_list_other,
                        quality_list_other,
                        patient_ids[i],
                        hours_to_use=NUM_HOURS_OTHER,
                        group="OTHER",
                    )
                    current_features = np.hstack((current_features, outcome_probabilities_torch_other))
                    current_feature_names = np.hstack((current_feature_names,torch_names_other))
                    outcome_probabilities_torch_other_aux.append(
                        outcome_probabilities_torch_other
                    )
                    outcome_flags_torch_other_aux.append(outcome_flags_torch_other)

        # Extract labels.
        features.append(current_features)
        feature_names.append(current_feature_names)
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        hospitals.append(hospital)
        recording_meta_infos.append(recording_infos)
        cpcs.append(current_cpc)
        patients.append(patient_ids[i])

    features = np.vstack(features)
    feature_names = np.vstack(feature_names)
    outcomes = np.vstack(outcomes)
    patients = np.vstack(patients)
    cpcs = np.vstack(cpcs)
    hospitals = np.vstack(hospitals)
    recording_meta_infos = np.vstack(recording_meta_infos)

    # Save the features and outcomes features.
    print("Saving features...")
    train_pd = pd.DataFrame(features, columns=feature_names[0])
    train_pd["patient_ids"] = patients
    train_pd["outcome"] = outcomes
    train_pd["cpc"] = cpcs
    train_pd["hospital"] = hospitals
    first_column = train_pd.pop("patient_ids")
    train_pd.insert(0, "patient_ids", first_column)
    for key in recording_meta_infos[0][0].keys():
        values_aux = list()
        for i in range(len(recording_meta_infos)):
            values_aux.append(recording_meta_infos[i][0][key])
        train_pd[key] = values_aux
    train_pd.to_csv(os.path.join(model_folder, "train_features.csv"), index=False)

    # Impute any missing features.
    if IMPUTE:
        print("Imputing features...")
        imputer = SimpleImputer(
            strategy=IMPUTE_METHOD, fill_value=IMPUTE_CONSTANT_VALUE
        ).fit(features)
        features = imputer.transform(features)
    else:
        imputer = None

    # Save the imputed features.
    print("Saving imputed features...")
    train_imputed_pd = pd.DataFrame(features, columns=feature_names[0])
    train_imputed_pd["patient_ids"] = patients
    train_imputed_pd["outcome"] = outcomes
    train_imputed_pd["cpc"] = cpcs
    train_imputed_pd["hospital"] = hospitals
    for key in recording_meta_infos[0][0].keys():
        values_aux = list()
        for i in range(len(recording_meta_infos)):
            values_aux.append(recording_meta_infos[i][0][key])
        train_imputed_pd[key] = values_aux
    train_imputed_pd.to_csv(
        os.path.join(model_folder, "train_imputed_features.csv"), index=False
    )

    # Train the models.
    if ONLY_EEG_TORCH or ONLY_EEG_ROCKET:
        outcome_model = None
        cpc_model = None
    else:
        print("Start training challenge model...")
        if c_model == "rf":
            outcome_model = RandomForestClassifier(**params_rf, class_weight=CLASS_WEIGHT)
            cpc_model = RandomForestRegressor(**params_rf)
        elif c_model == "xgb":
            outcome_model = xgb.XGBClassifier(**params_xgb, class_weight=CLASS_WEIGHT)
            cpc_model = xgb.XGBRegressor(**params_xgb)
        outcome_model.fit(features, outcomes.ravel())
        cpc_model.fit(features, cpcs.ravel())
        print("Finished training challenge model.")

        # Plot and save feature importance.
        n = 120
        feature_importance = outcome_model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        if len(sorted_idx) > n:
            # Create a list of names of the features not being plotted
            not_plotted_features = [feature_names[0][i] for i in sorted_idx[:-n]]
            not_plotted_features = list(map(str, not_plotted_features))
            #print("Features not being plotted:", not_plotted_features)
            # Save not_plotted_features as a JSON file
            with open(os.path.join(model_folder, "not_plotted_features.json"), 'w') as f:
                json.dump(not_plotted_features, f)
            # Adjust indices to plot only the top n features
            sorted_idx = sorted_idx[-n:]
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        plt.figure(figsize=(12, min(n / 2, 100)))
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(feature_names[0])[sorted_idx])
        plt.xlabel("Relative Importance")
        plt.title("Variable Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, "feature_importance.png"))
        plt.close()

    # Save the models.
    save_challenge_model(
        model_folder,
        imputer,
        outcome_model,
        cpc_model,
        rocket_model,
        rocket_transform,
        torch_model_eeg=torch_model_eeg,
        torch_model_ecg=torch_model_ecg,
        torch_model_ref=torch_model_ref,
        torch_model_other=torch_model_other,
    )

    if verbose >= 1:
        print(f"Done after {round((time.time()-start_time)/60,4)} min.")


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    model_folder = model_folder.lower()
    filename = os.path.join(model_folder, "models.sav")
    model = joblib.load(filename)
    file_path_eeg = os.path.join(model_folder, "eeg", "checkpoint.pth")
    file_path_ecg = os.path.join(model_folder, "ecg", "checkpoint.pth")
    file_path_ref = os.path.join(model_folder, "ref", "checkpoint.pth")
    file_path_other = os.path.join(model_folder, "other", "checkpoint.pth")
    if USE_ROCKET:
        model["rocket_eeg"] = mlflow_sktime.load_model(model_uri=os.path.join(model_folder, "eeg", "rocket"))
        model["rocket_model_eeg"] = mlflow_sktime.load_model(model_uri=os.path.join(model_folder, "eeg", "rocket_model"))
    else:
        model["rocket_model_eeg"] = None
        model["rocket_eeg"] = None
    if USE_TORCH:
        model["torch_model_eeg"] = load_last_pt_ckpt(
            file_path_eeg, channel_size=len(EEG_CHANNELS)
        )
        if USE_ECG:
            model["torch_model_ecg"] = load_last_pt_ckpt(
                file_path_ecg, channel_size=len(ECG_CHANNELS)
            )
        else:
            model["torch_model_ecg"] = None
        if USE_REF:
            model["torch_model_ref"] = load_last_pt_ckpt(
                file_path_ref, channel_size=len(REF_CHANNELS)
            )
        else:
            model["torch_model_ref"] = None
        if USE_OTHER:
            model["torch_model_other"] = load_last_pt_ckpt(
                file_path_other, channel_size=len(OTHER_CHANNELS)
            )
        else:
            model["torch_model_other"] = None
    else:
        model["torch_model_eeg"] = None
        model["torch_model_ecg"] = None
        model["torch_model_ref"] = None
        model["torch_model_other"] = None
    return model


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose, return_eeg_torch_probs=False):
    imputer = models["imputer"]
    outcome_model = models["outcome_model"]
    cpc_model = models["cpc_model"]
    rocket_model = models["rocket_model_eeg"]
    rocket = models["rocket_eeg"]
    torch_model_eeg = models["torch_model_eeg"]
    torch_model_ecg = models["torch_model_ecg"]
    torch_model_ref = models["torch_model_ref"]
    torch_model_other = models["torch_model_other"]

    # Load data.
    if verbose >= 2:
        print("Loading normal features...")
    features, _, _, _ = get_features(data_folder, patient_id)

    if USE_ROCKET:
        data_set_eeg = RecordingsDataset(
            data_folder,
            patient_ids=[patient_id],
            device="cpu",
            load_labels=False,
            group="EEG",
            hours_to_use=NUM_HOURS_EEG,
            raw=True
        )
        data_loader_eeg = DataLoader(
            data_set_eeg,
            batch_size=10,
            num_workers=PARAMS_DEVICE["num_workers"],
            shuffle=False,
        )
        (
            output_list_rocket_eeg,
            patient_id_list_rocket_eeg,
            hour_list_rocket_eeg,
            quality_list_rocket_eeg,
        ) = rocket_prediction(rocket, rocket_model, data_loader_eeg)
        (
            outcome_probabilities_rocket_eeg,
            outcome_flags_rocket_eeg,
            rocket_names_eeg,
        ) = rocket_predictions_for_patient(
            output_list_rocket_eeg,
            patient_id_list_rocket_eeg,
            hour_list_rocket_eeg,
            quality_list_rocket_eeg,
            patient_id,
            hours_to_use=NUM_HOURS_EEG,
            group="EEG",
        )
        features = np.hstack((features, outcome_probabilities_rocket_eeg))
    # Torch prediction
    if USE_TORCH:
        if verbose >= 2:
            print("Predicting torch features...")
        if USE_GPU:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            device = torch.device("cpu")
        data_set_eeg = RecordingsDataset(
            data_folder,
            patient_ids=[patient_id],
            device=device,
            load_labels=False,
            group="EEG",
            hours_to_use=NUM_HOURS_EEG,
        )
        data_loader_eeg = DataLoader(
            data_set_eeg,
            batch_size=1,
            num_workers=PARAMS_DEVICE["num_workers"],
            shuffle=False,
        )
        (
            output_list_eeg,
            patient_id_list_eeg,
            hour_list_eeg,
            quality_list_eeg,
        ) = torch_prediction(torch_model_eeg, data_loader_eeg, device)
        (
            outcome_probabilities_torch_eeg,
            outcome_flags_torch_eeg,
            torch_names_eeg
        ) = torch_predictions_for_patient(
            output_list_eeg,
            patient_id_list_eeg,
            hour_list_eeg,
            quality_list_eeg,
            patient_id,
            hours_to_use=NUM_HOURS_EEG,
            group="EEG",
        )
        features = np.hstack((features, outcome_probabilities_torch_eeg))
        if USE_ECG:
            data_set_ecg = RecordingsDataset(
                data_folder,
                patient_ids=[patient_id],
                device=device,
                load_labels=False,
                group="ECG",
                hours_to_use=NUM_HOURS_ECG,
            )
            data_loader_ecg = DataLoader(
                data_set_ecg,
                batch_size=1,
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
            )
            (
                output_list_ecg,
                patient_id_list_ecg,
                hour_list_ecg,
                quality_list_ecg,
            ) = torch_prediction(torch_model_ecg, data_loader_ecg, device)
            (
                outcome_probabilities_torch_ecg,
                outcome_flags_torch_ecg,
                torch_names_ecg
            ) = torch_predictions_for_patient(
                output_list_ecg,
                patient_id_list_ecg,
                hour_list_ecg,
                quality_list_ecg,
                patient_id,
                hours_to_use=NUM_HOURS_ECG,
                group="ECG",
            )
            features = np.hstack((features, outcome_probabilities_torch_ecg))
        if USE_REF:
            data_set_ref = RecordingsDataset(
                data_folder,
                patient_ids=[patient_id],
                device=device,
                load_labels=False,
                group="REF",
                hours_to_use=NUM_HOURS_REF,
            )
            data_loader_ref = DataLoader(
                data_set_ref,
                batch_size=1,
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
            )
            (
                output_list_ref,
                patient_id_list_ref,
                hour_list_ref,
                quality_list_ref,
            ) = torch_prediction(torch_model_ref, data_loader_ref, device)
            (
                outcome_probabilities_torch_ref,
                outcome_flags_torch_ref,
                torch_names_ref
            ) = torch_predictions_for_patient(
                output_list_ref,
                patient_id_list_ref,
                hour_list_ref,
                quality_list_ref,
                patient_id,
                hours_to_use=NUM_HOURS_REF,
                group="REF",
            )
            features = np.hstack((features, outcome_probabilities_torch_ref))
        if USE_OTHER:
            data_set_other = RecordingsDataset(
                data_folder,
                patient_ids=[patient_id],
                device=device,
                load_labels=False,
                group="OTHER",
                hours_to_use=NUM_HOURS_OTHER,
            )
            data_loader_other = DataLoader(
                data_set_other,
                batch_size=1,
                num_workers=PARAMS_DEVICE["num_workers"],
                shuffle=False,
            )
            (
                output_list_other,
                patient_id_list_other,
                hour_list_other,
                quality_list_other,
            ) = torch_prediction(torch_model_other, data_loader_other, device)
            (
                outcome_probabilities_torch_other,
                outcome_flags_torch_other,
                torch_names_other
            ) = torch_predictions_for_patient(
                output_list_other,
                patient_id_list_other,
                hour_list_other,
                quality_list_other,
                patient_id,
                hours_to_use=NUM_HOURS_OTHER,
                group="OTHER",
            )
            features = np.hstack((features, outcome_probabilities_torch_other))
    else:
        outcome_probabilities_torch_eeg = float('nan')

    if ONLY_EEG_TORCH:
        features = np.array(outcome_probabilities_torch_eeg)
    elif ONLY_EEG_ROCKET:
        features = np.array(outcome_probabilities_rocket_eeg)

    # Impute missing data.
    features = features.reshape(1, -1)
    if imputer is not None:
        features = imputer.transform(features)

    # Apply models to features.
    if ONLY_EEG_TORCH:
        outcome = features[0]
        outcome_probability = features[0]
        cpc = [1 if outcome_probability < 0.5 else 5]
    elif ONLY_EEG_ROCKET:
        outcome = features[0]
        outcome_probability = features[0]
        cpc = [1 if outcome_probability < 0.5 else 5]
    else:
        if verbose >= 2:
            print("Applying models...")
        outcome = outcome_model.predict(features)[0]
        outcome_probability = outcome_model.predict_proba(features)[0, 1]
        # outcome_probability = agg_outcome_probability_torch
        # outcome = 1 if outcome_probability > 0.5 else 0
        cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    if return_eeg_torch_probs:
        return outcome, outcome_probability, cpc, outcome_probabilities_torch_eeg
    else:
        return outcome, outcome_probability, cpc


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
def get_quality(string):
    start_time_sec = convert_hours_minutes_seconds_to_seconds(*get_start_time(string))
    end_time_sec = convert_hours_minutes_seconds_to_seconds(*get_end_time(string))
    quality = (end_time_sec - start_time_sec) / 3600
    return quality


def get_hour(string):
    hour_start = get_start_time(string)[0]
    return hour_start


def find_recording_files(data_folder, patient_id, group="", verbose=1):
    record_names = list()
    patient_folder = os.path.join(data_folder, patient_id)
    files_skipped = list()
    for file_name in sorted(os.listdir(patient_folder)):
        if not file_name.startswith(".") and file_name.endswith(f"{group}.hea"):
            hea_file = load_text_file(os.path.join(patient_folder, file_name))
            quality = get_quality(hea_file)
            seconds_available = quality * 3600
            if seconds_available >= MIN_SIGNAL_LENGTH:
                root, ext = os.path.splitext(file_name)
                record_name = "_".join(root.split("_")[:-1])
                record_names.append(record_name)
            else:
                files_skipped.append(file_name)

    if verbose >= 1:
        if (len(record_names) < NUM_HOURS_TO_USE) or (verbose >= 2):
            if len(files_skipped) > 0:
                print(
                    f"Skipped {len(files_skipped)} files for patient {patient_id} because they were too short."
                )
            else:
                print("No files skipped.")
            print(
                f"Found {len(record_names)} files for patient {patient_id} for group {group}."
            )

    return sorted(record_names)


def get_last_chkpt(model_folder):
    model_folder = model_folder.lower()

    if USE_BEST_MODEL:
        chkp_name = "checkpoint_best.pth"
        print("Using best model.")
    else:
        chkp_name = "checkpoint_last.pth"
        print("Using last model.")

    if os.path.exists(model_folder):
        if os.path.isfile(f"{model_folder}/{chkp_name}"):
            checkpoint_path = f"{model_folder}/{chkp_name}"
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None

    if checkpoint_path is not None:
        print("Resuming from checkpoint: ", checkpoint_path)
    else:
        print("No checkpoint found. Starting from scratch.")

    return checkpoint_path


def rocket_prediction(trf, clf, data_loader):
    output_list = []
    patient_id_list = []
    hour_list = []
    quality_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        data, features, targets, ids, hours, qualities = (
            batch["signal"],
            batch["features"],
            batch["label"],
            batch["id"],
            batch["hour"],
            batch["quality"],
        )
        inputs = trf.transform(data.cpu().detach().numpy())
        outputs = clf.predict(inputs)
        output_list = output_list + outputs.tolist()
        patient_id_list = patient_id_list + ids
        hour_list = hour_list + list(hours.cpu().detach().numpy())
        quality_list = quality_list + list(qualities.cpu().detach().numpy())
    return output_list, patient_id_list, hour_list, quality_list


def torch_prediction(model, data_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        output_list = []
        patient_id_list = []
        hour_list = []
        quality_list = []
        for _, batch in enumerate(tqdm(data_loader)):
            data, features, targets, ids, hours, qualities = (
                batch["image"],
                batch["features"],
                batch["label"],
                batch["id"],
                batch["hour"],
                batch["quality"],
            )
            data = data.to(device)
            features = features.to(device)
            outputs = model(data, features)
            outputs = torch.sigmoid(outputs)
            output_list = output_list + outputs.cpu().numpy().tolist()
            patient_id_list = patient_id_list + ids
            hour_list = hour_list + list(hours.cpu().detach().numpy())
            quality_list = quality_list + list(qualities.cpu().detach().numpy())
    return output_list, patient_id_list, hour_list, quality_list

def rocket_predictions_for_patient(
    output_list,
    patient_id_list,
    hour_list,
    quality_list,
    patient_id,
    max_hours=72,
    min_quality=0,
    num_signals=None,
    hours_to_use=None,
    group="",
):
    # Get the predictions for the patient
    patient_mask = np.array(
        [True if p == patient_id else False for p in patient_id_list]
    )
    if len(patient_mask) == 0:
        outcome_probabilities_rocket = np.array([])
        hours_patients = np.array([])
    else:
        outcome_probabilities_rocket = np.array(output_list)[patient_mask]
        hours_patients = np.array(hour_list)[patient_mask].astype(int).tolist()
        outcome_probabilities_rocket = [i for i in outcome_probabilities_rocket]
    outcome_flags_rocket = [
        1 if hour in hours_patients else 0 for hour in range(max_hours)
    ]
    # Get the hours to use for naming
    if hours_to_use >= 0:
        count_aux = list(range(hours_to_use))
    else:
        count_aux = list(range(hours_to_use, 0))
    if len(outcome_probabilities_rocket) < abs(hours_to_use):
        len_diff = abs(hours_to_use) - len(outcome_probabilities_rocket)
        aux_list = [np.nan] * len_diff
        if len(aux_list) == abs(hours_to_use):
            outcome_probabilities_rocket = aux_list
        else:
            if hours_to_use < 0:
                outcome_probabilities_rocket = aux_list + outcome_probabilities_rocket
            else:
                outcome_probabilities_rocket = outcome_probabilities_rocket + aux_list

    # Impute the missing values
    if IMPUTE:
        outcome_probabilities_rocket_imputed = (
            pd.Series(outcome_probabilities_rocket, dtype=object).bfill().tolist()
        )
        outcome_probabilities_rocket_imputed = (
            pd.Series(outcome_probabilities_rocket_imputed, dtype=object).ffill().tolist()
        )
    else:
        outcome_probabilities_rocket_imputed = pd.Series(
            outcome_probabilities_rocket, dtype=object
        ).tolist()
    if USE_AGGREGATION:
        count_aux = ['mean']
        outcome_probabilities_rocket_imputed = sum(outcome_probabilities_rocket_imputed)/len(outcome_probabilities_rocket_imputed)
    rocket_names = [f"prob_{group}_rocket_{i}" for i in count_aux]

    return (
        outcome_probabilities_rocket_imputed,
        outcome_flags_rocket,
        rocket_names
    )
    

def torch_predictions_for_patient(
    output_list,
    patient_id_list,
    hour_list,
    quality_list,
    patient_id,
    max_hours=72,
    min_quality=0,
    num_signals=None,
    hours_to_use=None,
    group="",
):
    # Get the predictions for the patient
    patient_mask = np.array(
        [True if p == patient_id else False for p in patient_id_list]
    )
    if len(patient_mask) == 0:
        outcome_probabilities_torch = np.array([])
        hours_patients = np.array([])
    else:
        outcome_probabilities_torch = np.array(output_list)[patient_mask]
        hours_patients = np.array(hour_list)[patient_mask].astype(int).tolist()

        if len(outcome_probabilities_torch[0]) == 1:
            outcome_probabilities_torch = [i[0] for i in outcome_probabilities_torch]
        else:
            raise ValueError(
                "The torch model should only predict one value per patient."
            )
    outcome_flags_torch = [
        1 if hour in hours_patients else 0 for hour in range(max_hours)
    ]

    # Aggregate the probabilities
    if USE_AGGREGATION:
        if AGGREGATION_METHOD == "voting":
            agg_outcome_probability_torch = [1 if v > DECISION_THRESHOLD else 0 for v in outcome_probabilities_torch]
            needed_votes = int(np.ceil(len(agg_outcome_probability_torch) * VOTING_POS_MAJORITY_THRESHOLD))
            agg_outcome_probability_torch = 1 if sum(agg_outcome_probability_torch) >= needed_votes else 0
            outcome_probabilities_torch_imputed = [agg_outcome_probability_torch]
            count_aux = ["voted"]
        elif AGGREGATION_METHOD == "weighted_average":
            weights = range(len(outcome_probabilities_torch))
            agg_outcome_probability_torch = [
                p * w for p, w in zip(outcome_probabilities_torch, weights) if not np.isnan(p)
            ]
            weight_sum = sum(
                [w for p, w in zip(outcome_probabilities_torch, weights) if not np.isnan(p)]
            )
            if weight_sum == 0:
                agg_outcome_probability_torch = 0
            else:
                agg_outcome_probability_torch = sum(agg_outcome_probability_torch) / weight_sum
            outcome_probabilities_torch_imputed = [agg_outcome_probability_torch]
            count_aux = ["weighted_average"]
    else:
        # Get the hours to use for naming
        if hours_to_use >= 0:
            count_aux = list(range(hours_to_use))
        else:
            count_aux = list(range(hours_to_use, 0))

        # Fill in the missing hours
        if len(outcome_probabilities_torch) < abs(hours_to_use):
            len_diff = abs(hours_to_use) - len(outcome_probabilities_torch)
            aux_list = [np.nan] * len_diff
            if len(aux_list) == abs(hours_to_use):
                outcome_probabilities_torch = aux_list
            else:
                if hours_to_use < 0:
                    outcome_probabilities_torch = aux_list + outcome_probabilities_torch
                else:
                    outcome_probabilities_torch = outcome_probabilities_torch + aux_list

        # Impute the missing values
        if IMPUTE:
            outcome_probabilities_torch_imputed = (
                pd.Series(outcome_probabilities_torch, dtype=object).bfill().tolist()
            )
            outcome_probabilities_torch_imputed = (
                pd.Series(outcome_probabilities_torch_imputed, dtype=object).ffill().tolist()
            )
        else:
            outcome_probabilities_torch_imputed = pd.Series(
                outcome_probabilities_torch, dtype=object
            ).tolist()
        
        # Find the median of the signals where outcome_flags_torch is 1 and only keep the num_signals around the median
        if num_signals is not None:
            if num_signals > max_hours:
                raise ValueError("num_signals should be smaller than max_hours.")
            median = np.median(np.where(np.array(outcome_flags_torch) == 1)[0])
            lower_bound = int(median - num_signals / 2)
            upper_bound = int(median + num_signals / 2)
            if lower_bound < 0:
                upper_bound = upper_bound + abs(lower_bound)
                lower_bound = 0
            if upper_bound > max_hours:
                lower_bound = lower_bound - (upper_bound - max_hours)
                upper_bound = max_hours
            outcome_probabilities_torch_imputed = outcome_probabilities_torch_imputed[
                lower_bound:upper_bound
            ]
            outcome_flags_torch = outcome_flags_torch[lower_bound:upper_bound]
            count_aux = list(range(lower_bound, upper_bound))

    torch_names = [f"prob_{group}_torch_{i}" for i in count_aux]

    return (
        outcome_probabilities_torch_imputed,
        outcome_flags_torch,
        torch_names
    )


def get_tv_model(
    model_name="densenet121",
    num_classes=1,
    batch_size=64,
    d_size=500,
    pretrained=False,
    channel_size=3,
    additional_features=0,
):
    model = TorchvisionModel(
        model_name=model_name,
        num_classes=num_classes,
        print_freq=250,
        batch_size=batch_size,
        d_size=d_size,
        pretrained=pretrained,
        channel_size=channel_size,
        additional_features=additional_features,
    )
    return model


# Load last checkpoint
def load_last_pt_ckpt(ckpt_path, channel_size):
    if os.path.isfile(ckpt_path):
        if USE_GPU:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            device = torch.device("cpu")
        print(f"Loading checkpoint from {ckpt_path}")
        if "pth" in ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dic = checkpoint["model"]
            if "additional_layer.weight" in state_dic.keys():
                additional_features = len(state_dic["additional_layer.weight"])
            else:
                additional_features = 0
            model = get_tv_model(channel_size=channel_size, additional_features=additional_features)
            model.load_state_dict(state_dic)
        elif "ckpt" in ckpt_path:
            model = get_tv_model(channel_size=channel_size)
            model = model.load_from_checkpoint(ckpt_path)
        return model
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")


# Save your trained model.
def save_challenge_model(
    model_folder, imputer, outcome_model, cpc_model, rocket_model, rocket_transform, **torch_models
):
    model_folder = model_folder.lower()
    d = {"imputer": imputer, "outcome_model": outcome_model, "cpc_model": cpc_model}
    filename = os.path.join(model_folder, "models.sav")
    joblib.dump(d, filename, protocol=0)
    if rocket_model is not None:
        # Save the rocket model, if path does not exist, create it, if file already exists, overwrite it.
        if os.path.exists(os.path.join(model_folder, "eeg", "rocket")):
            os.rmdir(os.path.join(model_folder, "eeg", "rocket"))
            print("Removed old rocket model.")
        if not os.path.exists(os.path.join(model_folder, "eeg")):
            os.makedirs(os.path.join(model_folder, "eeg"))
        mlflow_sktime.save_model(sktime_model=rocket_transform, path=os.path.join(model_folder, "eeg", "rocket"))
        mlflow_sktime.save_model(sktime_model=rocket_model, path=os.path.join(model_folder, "eeg", "rocket_model"))
    for name, torch_model in torch_models.items():
        if torch_model is not None:
            torch_model_folder = os.path.join(model_folder, name.split("_")[-1])
            if not os.path.exists(torch_model_folder):
                os.makedirs(torch_model_folder)
            file_path = os.path.join(torch_model_folder, "checkpoint.pth")
            torch.save({"model": torch_model.state_dict()}, file_path)


# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.5, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if (
        utility_frequency is not None
        and passband[0] <= utility_frequency <= passband[1]
    ):
        data = mne.filter.notch_filter(
            data, sampling_frequency, utility_frequency, n_jobs=1, verbose="error"
        )

    # Apply a bandpass filter.
    data = mne.filter.filter_data(
        data, sampling_frequency, passband[0], passband[1], n_jobs=1, verbose="error"
    )

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency


# Load recording data.
def get_recording_features(
    recording_ids, recording_id_to_use, data_folder, patient_id, group, channels_to_use
):
    if group == "EEG":
        if BIPOLAR_MONTAGES is not None:
            channel_length = len(BIPOLAR_MONTAGES)
        else:
            channel_length = len(channels_to_use)
        dummy_channels = channel_length * 4
        recording_feature_names = np.array(
            (
                np.array(
                    [
                        f"delta_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"theta_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"alpha_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"beta_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
            )
        ).T.flatten()
    elif (group == "ECG") or (group == "REF") or (group == "OTHER"):
        channel_length = len(channels_to_use)
        dummy_channels = channel_length * 2
        recording_feature_names = np.array(
            (
                np.array(
                    [
                        f"{group}_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"{group}_std_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
            )
        ).T.flatten()
    else:
        raise ValueError("Group should be either EEG, ECG, REF or OTHER")

    if len(recording_ids) > 0:
        if abs(recording_id_to_use) <= len(recording_ids):
            recording_id = recording_ids[recording_id_to_use]
            recording_location = os.path.join(
                data_folder, patient_id, "{}_{}".format(recording_id, group)
            )
            if os.path.exists(recording_location + ".hea"):
                data, channels, sampling_frequency = load_recording_data_wrapper(
                    recording_location, channels_to_use
                )
                hea_file = load_text_file(recording_location + ".hea")
                utility_frequency = get_utility_frequency(hea_file)
                quality = get_quality(hea_file)
                hour = get_hour(hea_file)
                if (
                    all(channel in channels for channel in channels_to_use)
                    or group != "EEG"
                ):
                    data, channels = reduce_channels(data, channels, channels_to_use)
                    data, sampling_frequency = preprocess_data(
                        data, sampling_frequency, utility_frequency
                    )
                    if group == "EEG":
                        if BIPOLAR_MONTAGES is not None:
                            data = np.array(
                                [
                                    data[channels.index(montage[0]), :]
                                    - data[channels.index(montage[1]), :]
                                    for montage in BIPOLAR_MONTAGES
                                ]
                            )
                        recording_features = get_eeg_features(data, sampling_frequency)
                    elif (group == "ECG") or (group == "REF") or (group == "OTHER"):
                        features = get_ecg_features(data)
                        recording_features = expand_channels(
                            features, channels, channels_to_use
                        ).flatten()
                    else:
                        raise NotImplementedError(f"Group {group} not implemented.")
                else:
                    print(
                        f"For patient {patient_id} recording {recording_id} the channels {channels_to_use} are not all available. Only {channels} are available."
                    )
                    recording_features = float("nan") * np.ones(
                        dummy_channels
                    )  # 2 bipolar channels * 4 features / channel
            else:
                recording_features = float("nan") * np.ones(
                    dummy_channels
                )  # 2 bipolar channels * 4 features / channel
                sampling_frequency = (
                    utility_frequency
                ) = channels = quality = hour = np.nan
        else:
            recording_features = float("nan") * np.ones(
                dummy_channels
            )  # 2 bipolar channels * 4 features / channel
            sampling_frequency = (
                utility_frequency
            ) = channels = recording_id = quality = hour = np.nan
    else:
        recording_features = float("nan") * np.ones(
            dummy_channels
        )  # 2 bipolar channels * 4 features / channel
        sampling_frequency = (
            utility_frequency
        ) = channels = recording_id = quality = hour = np.nan

    # Aggregate over channels
    if AGG_OVER_CHANNELS:
        recording_feature_group_names = [f'{f.split("_c_")[0]}_hour_{f.split("_c_")[-1].split("hour_")[-1]}' for f in recording_feature_names]
        recording_features, recording_feature_names = aggregate_features(recording_features, recording_feature_names, recording_feature_group_names)
        channels = [f"Agg_over_{int(len(recording_feature_group_names)/len(recording_feature_names))}_channels"]

    return (
        recording_features,
        recording_feature_names,
        sampling_frequency,
        utility_frequency,
        channels,
        recording_id,
        quality,
        hour,
    )


def aggregate_features(recording_features, recording_feature_names, recording_feature_group_names):
    unique_groups = np.unique(recording_feature_group_names)
    recording_features_agg = np.zeros(len(unique_groups))
    for i, group in enumerate(unique_groups):
        aux_values = recording_features[[True if group == f else False for f in recording_feature_group_names]]
        if len(aux_values) == 0 or np.all(np.isnan(aux_values)):
            recording_features_agg[i] = float("nan")
        else:
            recording_features_agg[i] = np.nanmean(aux_values)
    recording_features = recording_features_agg
    recording_feature_names = unique_groups

    return recording_features, recording_feature_names


# Extract features from the data.
def get_features(data_folder, patient_id, return_as_dict=False, recording_features=True, normalize = False):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids_eeg = find_recording_files(data_folder, patient_id, "EEG")
    use_last_hours_eeg, hours_eeg, start_eeg = get_correct_hours(NUM_HOURS_EEG)
    recording_ids_ecg = find_recording_files(data_folder, patient_id, "ECG")
    use_last_hours_ecg, hours_ecg, start_ecg = get_correct_hours(NUM_HOURS_ECG)
    recording_ids_ref = find_recording_files(data_folder, patient_id, "REF")
    use_last_hours_ref, hours_ref, start_ref = get_correct_hours(NUM_HOURS_REF)
    recording_ids_other = find_recording_files(data_folder, patient_id, "OTHER")
    use_last_hours_other, hours_other, start_other = get_correct_hours(NUM_HOURS_OTHER)

    # Extract patient features.
    patient_features, patient_feature_names = get_patient_features(
        patient_metadata,
        recording_ids_eeg,
        recording_ids_ecg,
        recording_ids_ref,
        recording_ids_other,
        normalize = normalize
    )
    hospital = get_hospital(patient_metadata)

    # Extract recording features.
    feature_values = patient_features
    feature_names = patient_feature_names
    recording_infos = {}
    if recording_features:
        feature_types = ["EEG", "ECG", "REF", "OTHER"]
        use_flags = {"EEG": True, "ECG": USE_ECG, "REF": USE_REF, "OTHER": USE_OTHER}
        starts = {
            "EEG": start_eeg,
            "ECG": start_ecg,
            "REF": start_ref,
            "OTHER": start_other,
        }
        hours = {"EEG": hours_eeg, "ECG": hours_ecg, "REF": hours_ref, "OTHER": hours_other}
        use_last_hours = {
            "EEG": use_last_hours_eeg,
            "ECG": use_last_hours_ecg,
            "REF": use_last_hours_ref,
            "OTHER": use_last_hours_other,
        }
        recording_ids = {
            "EEG": recording_ids_eeg,
            "ECG": recording_ids_ecg,
            "REF": recording_ids_ref,
            "OTHER": recording_ids_other,
        }
        channels_to_use = {
            "EEG": EEG_CHANNELS,
            "ECG": ECG_CHANNELS,
            "REF": REF_CHANNELS,
            "OTHER": OTHER_CHANNELS,
        }
        for feature_type in feature_types:
            if use_flags[feature_type]:
                feature_data = process_recording_feature(
                    feature_type,
                    starts[feature_type],
                    hours[feature_type],
                    use_last_hours[feature_type],
                    recording_ids[feature_type],
                    channels_to_use[feature_type],
                    data_folder,
                    patient_id,
                )
                recording_infos.update(feature_data)
                feature_values = np.hstack(
                    (feature_values, np.hstack(feature_data[f"{feature_type}_features"]))
                )
                feature_names = np.hstack(
                    (
                        feature_names,
                        np.hstack(feature_data[f"{feature_type}_feature_names"]),
                    )
                )

    if return_as_dict:
        return {k: v for k, v in zip(feature_names, feature_values)}
    else:
        return feature_values, feature_names, hospital, recording_infos


# Extract patient features from the data.
def get_patient_features(
    data, recording_ids_eeg, recording_ids_ecg, recording_ids_ref, recording_ids_other, normalize = False
):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == "Female":
        female = 1
        male = 0
        other = 0
    elif sex == "Male":
        female = 0
        male = 1
        other = 0
    else:
        female = 0
        male = 0
        other = 1

    if len(recording_ids_eeg) > 0:
        last_eeg_hour = np.max([int(r.split("_")[-1]) for r in recording_ids_eeg])
    else:
        last_eeg_hour = np.nan

    # Get binary features wheather the different signals are available.
    eeg_available = 1 if len(recording_ids_eeg) > 0 else 0
    ecg_available = 1 if len(recording_ids_ecg) > 0 else 0
    ref_available = 1 if len(recording_ids_ref) > 0 else 0
    other_available = 1 if len(recording_ids_other) > 0 else 0

    # Normalize
    if normalize:
        age = age / 100
        rosc = rosc / 200
        ttm = ttm / 36
        last_eeg_hour = last_eeg_hour / 72

    features = np.array(
        (
            age,
            female,
            male,
            #other,
            rosc,
            ohca,
            shockable_rhythm,
            ttm,
            last_eeg_hour,
            ecg_available,
            ref_available,
            other_available,
        )
    )
    feature_names = [
        "age",
        "female",
        "male",
        #"other",
        "rosc",
        "ohca",
        "shockable_rhythm",
        "ttm",
        "last_eeg_hour",
        "ecg_available",
        "ref_available",
        "other_available",
    ]

    return features, feature_names


# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=0.5, fmax=4.0, verbose=False
        )
        theta_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=4.0, fmax=8.0, verbose=False
        )
        alpha_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=8.0, fmax=12.0, verbose=False
        )
        beta_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False
        )

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean = np.nanmean(beta_psd, axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float(
            "nan"
        ) * np.ones(num_channels)
    features = np.array(
        (delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)
    ).T
    features = features.flatten()

    return features


# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std = float("nan") * np.ones(num_channels)
    else:
        mean = float("nan") * np.ones(num_channels)
        std = float("nan") * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features


class RecordingsDataset(Dataset):
    def __init__(
        self,
        data_folder,
        patient_ids,
        device,
        group="EEG",
        load_labels: bool = True,
        hours_to_use: int = None,
        raw: bool = False,
    ):

        self.raw = raw
        self._precision = torch.float32
        self.hours_to_use = hours_to_use
        self.group = group
        if self.group == "EEG":
            self.channels_to_use = EEG_CHANNELS
        elif self.group == "ECG":
            self.channels_to_use = ECG_CHANNELS
        elif self.group == "REF":
            self.channels_to_use = REF_CHANNELS
        elif self.group == "OTHER":
            self.channels_to_use = OTHER_CHANNELS
        else:
            raise NotImplementedError(f"Group {self.group} not implemented.")

        # Load labels and features
        recording_locations_list = list()
        patient_ids_list = list()
        labels_list = list()
        features_list = list()
        for patient_id in patient_ids:
            patient_metadata = load_challenge_data(data_folder, patient_id)
            if INFUSE_STATIC_FEATURES:
                (
                    current_features,
                    current_feature_names,
                    hospital,
                    recording_infos,
                ) = get_features(data_folder, patient_id, recording_features = False, normalize = True)
            else:
                current_features = np.nan
            recording_ids = find_recording_files(
                data_folder, patient_id, self.group, verbose=0
            )
            if self.hours_to_use is not None:
                if abs(self.hours_to_use) < len(recording_ids):
                    if self.hours_to_use > 0:
                        recording_ids = recording_ids[: self.hours_to_use]
                    else:
                        recording_ids = recording_ids[self.hours_to_use :]
            if load_labels:
                current_outcome = get_outcome(patient_metadata)
            else:
                current_outcome = 0
            for recording_id in recording_ids:
                if not is_nan(recording_id):
                    recording_location_aux = os.path.join(
                        data_folder,
                        patient_id,
                        "{}_{}".format(recording_id, self.group),
                    )
                    if os.path.exists(recording_location_aux + ".hea"):
                        recording_locations_list.append(recording_location_aux)
                        patient_ids_list.append(patient_id)
                        labels_list.append(current_outcome)
                        features_list.append(current_features)

        self.recording_locations = recording_locations_list
        self.patient_ids = patient_ids_list
        self.labels = labels_list
        self.features = features_list
        if INFUSE_STATIC_FEATURES:
            self.num_additional_features = len(current_features)
        else:
            self.num_additional_features = 0
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load the data.
        try:
            (
                signal_data,
                signal_channels,
                sampling_frequency,
            ) = load_recording_data_wrapper(
                self.recording_locations[idx], self.channels_to_use
            )
        except Exception as e:
            print("Error loading {}".format(self.recording_locations[idx]))
            raise e
        hea_file = load_text_file(self.recording_locations[idx] + ".hea")
        utility_frequency = get_utility_frequency(hea_file)
        signal_data, signal_channels = reduce_channels(
            signal_data, signal_channels, self.channels_to_use
        )

        # Preprocess the data.
        sampling_frequency_old = sampling_frequency
        signal_data, sampling_frequency = preprocess_data(
            signal_data, sampling_frequency, utility_frequency
        )

        # Get the other information.
        id = self.patient_ids[idx]
        hour = get_hour(hea_file)
        quality = get_quality(hea_file)

        # Get the label
        label = self.labels[idx]
        label = torch.from_numpy(np.array(label).astype(np.float32)).to(self._precision)

        # Get the static features.
        static_features = self.features[idx]
        static_features = np.nan_to_num(static_features, nan=-1)
        static_features = torch.from_numpy(np.array(static_features).astype(np.float32)).to(self._precision)

        if self.raw:
            return_dict = {
                "signal": signal_data,
                "features": static_features,
                "label": self.labels[idx],
                "id": id,
                "hour": hour,
                "quality": quality,
            }

            return return_dict

        points_full_hour = 60 * 60 * sampling_frequency
        available_points = signal_data.shape[1] / points_full_hour
        target_size = 901
        hop_length = max(int(round(signal_data.shape[1] / target_size, 0)), 1)

        # Get the spectrograms.
        n_fft = 2**10
        if signal_data.shape[1] < n_fft:
            pad_length = n_fft - signal_data.shape[1]
            padded_signal_data = np.pad(signal_data, ((0, 0), (0, pad_length)))
            print(
                f"Padding signal {self.recording_locations[idx]} of length {signal_data.shape[1]} with {pad_length} zeros for n_fft {n_fft}"
            )
        else:
            padded_signal_data = signal_data
        spectrograms = librosa.feature.melspectrogram(
            y=padded_signal_data,
            sr=sampling_frequency,
            n_mels=224,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        spectrograms = torch.from_numpy(spectrograms.astype(np.float32))
        spectrograms = nn.functional.normalize(spectrograms).to(self._precision)
        spectrograms = spectrograms.unsqueeze(0)
        spectrograms_resized = F.interpolate(
            spectrograms,
            size=(spectrograms.shape[2], target_size),
            mode="bilinear",
            align_corners=False,
        )
        spectrograms = spectrograms_resized.squeeze(0)


        return_dict = {
            "image": spectrograms.to(self._precision),
            "features": static_features,
            "label": label.to(self._precision),
            "id": id,
            "hour": hour,
            "quality": quality,
        }

        return return_dict


class TorchvisionModel(torch.nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=2,
        classification="binary",
        print_freq=100,
        batch_size=10,
        d_size=500,
        pretrained=False,
        channel_size=3,
        additional_features=0,  # size of additional features
    ):
        super().__init__()
        self._d_size = d_size
        self._b_size = batch_size
        self._print_freq = print_freq
        self.model_name = model_name
        self.num_classes = num_classes
        self.classification = classification
        self.additional_features = additional_features
        self.model = eval(f"models.{model_name}()")
        if pretrained:
            print(f"Using pretrained {model_name} model")
            state_dict = torch.load("densenet121-a639ec97.pth")
            state_dict = self.update_densenet_keys(state_dict)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Using {model_name} model from scratch")
        self.model.features[0] = nn.Conv2d(
            channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # add a linear layer for additional features
        self.additional_layer = nn.Linear(additional_features, additional_features)

        if "resnet" in model_name.lower():
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features + additional_features, self.num_classes)
        elif "densenet" in model_name.lower():
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features + additional_features, self.num_classes)
        elif "vit_b_16" in model_name.lower():
            num_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(num_features + additional_features, self.num_classes)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def update_densenet_keys(self, state_dict):
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return state_dict
    
    def forward(self, x, additional_x=None, classify=True):
        if self.additional_features > 0:
            assert additional_x is not None, "additional_x cannot be None when using additional_features"
            assert additional_x.shape[1] == self.additional_features, "Wrong shape of additional features"
            features = self.model.features(x)  # Use the convolutional part only
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            additional_x = self.additional_layer(additional_x)
            features = torch.cat([features, additional_x], dim=1)  # concatenate along the feature dimension
            x = self.model.classifier(features)  # Now, use the classifier
        else:
            x = self.model.forward(x)
        
        return x

    def unpack_batch(self, batch):
        return batch["image"], batch["features"], batch["label"]

    def process_batch(self, batch):
        img, features, lab = self.unpack_batch(batch)
        out = self.forward(img, features)
        out = out.squeeze()
        if self.classification == "binary":
            prob = torch.sigmoid(out)
            if img.shape[0] == 1:
                prob = prob.unsqueeze(0)
            loss = F.binary_cross_entropy(prob, lab)
        else:
            raise NotImplementedError(
                f"Classification {self.classification} not implemented"
            )
        return loss

    def training_step(self, batch):
        loss = self.process_batch(batch)
        return loss

    def validation_step(self, batch):
        loss = self.process_batch(batch)
        return loss

    def test_step(self, batch):
        loss = self.process_batch(batch)
        return loss


@torch.no_grad()
def predict(data_loader, model, device) -> np.ndarray:
    # switch to evaluation mode
    model = model.to(device)
    model.eval()

    p_out = torch.FloatTensor().to(device)
    t_out = torch.FloatTensor().to(device)

    target_labels = np.empty(0)
    for iteration in tqdm(iter(data_loader)):
        # Get prediction
        output, target = predict_from_batches(iteration, model, device)
        p_out = torch.cat((p_out, output), 0)
        t_out = torch.cat((t_out, target), 0)

    # Get most likely class
    if model.multilabel:
        sigmoid = nn.Sigmoid()
        preds = sigmoid(p_out)
        preds_probs = preds.cpu().numpy()
        preds_labels = preds.gt(0.5).type(preds.dtype)
    else:
        softmax = nn.Softmax(dim=1)
        preds_probs = softmax(p_out).cpu().numpy()
        _, preds_labels = torch.max(p_out, 1)
    preds_labels = preds_labels.cpu().numpy()
    target_labels = t_out.cpu().numpy()

    print(
        f"Predicted {len(preds_probs)} observations with {len(preds_probs[0])} {'multilabel' if model.multilabel else 'multiclass'} labels"
    )

    return preds_probs, preds_labels, target_labels, p_out, preds


def predict_from_batches(iteration, model, device):
    # Extract data
    if type(iteration) == list:
        images = iteration[0]
        target = iteration[1]
    elif type(iteration) == dict:
        images = iteration["image"]
        features = iteration["features"]
        target = iteration["label"]
    else:
        raise ValueError("Something is wrong. __getitem__ must return list or dict")
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Get predictions
    with torch.no_grad():
        output = model(images, features, classify=True)

    return output, target


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_correct_hours(num_hours):
    if num_hours < 0:
        use_last_hours_recording = True
        hours_recording = -num_hours
        start_recording = 1
    elif num_hours > 0:
        use_last_hours_recording = False
        hours_recording = num_hours
        start_recording = 0
    else:
        raise ValueError("num_hours should be either positive or negative.")
    return use_last_hours_recording, hours_recording, start_recording


def process_recording_feature(
    feature_type,
    start,
    hours,
    use_last_hours,
    recording_ids,
    channels_to_use,
    data_folder,
    patient_id,
):
    feature_data = {
        f"{feature_type}_{item}": []
        for item in [
            "features",
            "feature_names",
            "sampling_frequency",
            "utility_frequency",
            "channels",
            "recording_id",
            "quality",
            "hour",
        ]
    }

    for h in range(start, hours + start):
        if use_last_hours:
            h_to_use = -h
        (
            features,
            feature_names,
            sampling_frequency,
            utility_frequency,
            channels,
            recording_id,
            quality,
            hour,
        ) = get_recording_features(
            recording_ids=recording_ids,
            recording_id_to_use=h_to_use,
            data_folder=data_folder,
            patient_id=patient_id,
            group=feature_type.upper(),
            channels_to_use=channels_to_use,
        )

        for item in feature_data.keys():
            feature_data[item].append(locals()[item.split("_", 1)[-1]])

    if AGG_OVER_TIME:
        feature_names_aux = np.array([item for row in feature_data[f"{feature_type}_feature_names"] for item in row])
        features_aux = np.array([item for row in feature_data[f"{feature_type}_features"] for item in row])
        recording_feature_group_names = [f'{f.split("_hour_")[0]}' for f in feature_names_aux]
        feature_data[f"{feature_type}_features"], feature_data[f"{feature_type}_feature_names"] = aggregate_features(features_aux, feature_names_aux, recording_feature_group_names)
        feature_data[f"{feature_type}_sampling_frequency"] = feature_data[f"{feature_type}_sampling_frequency"][0]
        feature_data[f"{feature_type}_utility_frequency"] = feature_data[f"{feature_type}_utility_frequency"][0]
        feature_data[f"{feature_type}_channels"] = feature_data[f"{feature_type}_channels"][0]
        value_aux = feature_data[f"{feature_type}_recording_id"][0]
        if isinstance(value_aux, str):
            feature_data[f"{feature_type}_recording_id"] = value_aux.split("_")[0]
        elif value_aux is np.nan:
            feature_data[f"{feature_type}_recording_id"] = value_aux
        else:
            raise ValueError(f"Unexpected value for recording_id: {value_aux}")
        feature_data[f"{feature_type}_hour"] = f'agg_from_{np.min(feature_data[f"{feature_type}_hour"])}_to_{np.max(feature_data[f"{feature_type}_hour"])}'
        feature_data[f"{feature_type}_quality"] = np.nanmean(feature_data[f"{feature_type}_quality"])

    return feature_data


def check_artifacts(
    epoch: np.ndarray,
    low_threshold: float = LOW_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
) -> bool:
    """
    Checks an EEG epoch for artifacts.

    Parameters
    ----------
    epoch: np.ndarray
        The EEG signal epoch as a 1D numpy array.
    low_threshold: float
        The lower limit for detecting extreme values.
    high_threshold: float
        The upper limit for detecting extreme values.

    Returns
    -------
    bool
        True if any type of artifact is detected, False otherwise.
    """
    # Flat signal
    if np.all(epoch == epoch[0]):
        return True

    # Extreme high or low values
    if np.max(epoch) > high_threshold or np.min(epoch) < low_threshold:
        return True

    # Muscle artifact (Assuming if the standard deviation of an epoch is high, it might be a muscle artifact)
    # if abs(np.std(epoch)) > 10 * abs(np.mean(epoch)):
    #    return True

    # Fast rising or decreasing signal amplitude (If the absolute difference between any two consecutive samples is above a threshold)
    # if np.any(np.abs(np.diff(epoch)) > 10 * np.mean(np.abs(np.diff(epoch)))):
    #    return True

    return False


def compute_score(
    signal: np.ndarray,
    signal_frequency: float,
    epoch_size: int,
    window_size: int,
    stride_length: int,
    high_threshold: float = None,
    low_threshold: float = None,
) -> Tuple[Dict, Dict]:
    """
    Computes a score for each window in the EEG signal.

    Parameters
    ----------
    signal: np.ndarray
        The EEG signal as a 2D numpy array of shape (channel, num_observations).
    signal_frequency: float
        The frequency of the signal in Hz.
    epoch_size: int
        The size of each epoch in seconds.
    window_size: int
        The size of each window in minutes.
    stride_length: int
        The stride length in minutes for moving the window.
    high_threshold: float
        The upper limit for detecting extreme values.
    low_threshold: float
        The lower limit for detecting extreme values.

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple of two dictionaries. The first dictionary contains the score for each window. The second dictionary contains the channels to be replaced for each window.
    """
    num_channels, num_observations = signal.shape
    epoch_samples = int(signal_frequency * epoch_size)
    window_samples = int(signal_frequency * window_size * 60)
    stride_samples = int(signal_frequency * stride_length * 60)
    scores = {}
    channels_to_replace = {}
    if window_samples > num_observations:
        window_samples = num_observations
    if stride_samples > num_observations:
        stride_samples = num_observations
    for start in range(0, num_observations - window_samples + 1, stride_samples):
        window = signal[:, start : start + window_samples]
        if epoch_samples > window_samples:
            epoch_samples = window_samples
        artifact_epochs = 0
        total_epochs = 0
        ignored_artifcat_epochs = 0
        channels_with_artifacts = []
        for epoch_start in range(0, window_samples - epoch_samples + 1, epoch_samples):
            no_channels_with_artifacts = 0
            artifact_epochs_old = artifact_epochs
            for channel in range(num_channels):
                epoch = window[channel, epoch_start : epoch_start + epoch_samples]
                if check_artifacts(epoch):
                    no_channels_with_artifacts += 1
                    channels_with_artifacts.append(channel)
                if no_channels_with_artifacts > NO_CHANNELS_W_ARTIFACT_TO_DISCARD_EPOCH:
                    artifact_epochs += 1
                    break  # if any channel in the epoch has artifact, consider the whole epoch contaminated
            if artifact_epochs == artifact_epochs_old:
                if no_channels_with_artifacts > 0:
                    ignored_artifcat_epochs += 1
            total_epochs += 1
        if len(set(channels_with_artifacts)) > NO_CHANNELS_W_ARTIFACT_TO_DISCARD_WINDOW:
            artifact_epochs = artifact_epochs + ignored_artifcat_epochs
            channels_with_artifacts = []
        score = 1 - artifact_epochs / total_epochs
        # Converting start sample index to time in seconds
        start_time = start / signal_frequency
        scores[start_time] = score
        channels_to_replace[start_time] = list(set(channels_with_artifacts))
    return scores, channels_to_replace


def keep_best_window(
    signal: np.ndarray,
    scores: Dict[int, float],
    signal_frequency: float,
    window_size: int,
    channels_to_replace: Dict[int, float],
    channels_to_use: list(),
) -> np.ndarray:
    """
    Keep only the window with the best score in the EEG signal and set all other samples to NaN.

    Parameters
    ----------
    signal: np.ndarray
        The EEG signal as a 2D numpy array of shape (channel, num_observations).
    scores: Dict[int, float]
        A dictionary where keys are the starting time (in seconds) of each window and values are the corresponding scores.
    signal_frequency: float
        The frequency of the signal in Hz.
    window_size: int
        The size of each window in minutes.
    channels_to_replace: Dict[int, float]
        A dictionary where keys are the starting time (in seconds) of each window and values are the corresponding channels to be replace.
    channels_to_use: list()
        A list of channels to use.

    Returns
    -------
    np.ndarray
        The EEG signal where only the window with the best score is kept and all other samples are set to NaN.
    """
    num_channels, num_observations = signal.shape
    window_samples = int(signal_frequency * window_size * 60)

    # Find the window with the best score
    best_start_time = get_max_key(scores)
    best_start_sample = int(best_start_time * signal_frequency)

    # Check if channels_to_replace contains for best_start_time channels that have to be replaced
    new_signal = signal.copy()
    if len(channels_to_replace[best_start_time]) > 0:
        # Replace the channels that have to be replaced with any random channel from channels_to_use that is not in channels_to_replace[best_start_time]
        for channel in channels_to_replace[best_start_time]:
            random_channel = random.choice(
                [
                    c
                    for c in range(signal.shape[0])
                    if c not in channels_to_replace[best_start_time]
                ]
            )
            new_signal[channel, :] = signal[random_channel, :]

    if best_start_sample + window_samples > num_observations:
        new_signal = new_signal[:, best_start_sample:]
    else:
        new_signal = new_signal[
            :, best_start_sample : best_start_sample + window_samples
        ]

    return new_signal


def load_recording_data_wrapper(record_name, channels_to_use):
    """
    Loads the EEG signal of a recording and applies artifact removal.
    """

    window_size = WINDOW_SIZE_FILTER  # minutes
    stride_length = STRIDE_SIZE_FILTER  # minutes
    epoch_size = EPOCH_SIZE_FILTER  # seconds

    signal_data, signal_channels, sampling_frequency = load_recording_data(record_name)

    # Remove the first and last x seconds of the recording to avoid edge effects.
    num_samples_to_remove = int(
        SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING * sampling_frequency
    )
    if num_samples_to_remove > 0:
        if signal_data.shape[1] > (
            2 * num_samples_to_remove + sampling_frequency * 60 * 5
        ):
            signal_data = signal_data[:, num_samples_to_remove:-num_samples_to_remove]

    if FILTER_SIGNALS:
        scores, channels_to_replace = compute_score(
            signal_data,
            signal_frequency=sampling_frequency,
            epoch_size=epoch_size,
            window_size=window_size,
            stride_length=stride_length,
        )
        signal_data_filtered = keep_best_window(
            signal=signal_data,
            scores=scores,
            signal_frequency=sampling_frequency,
            window_size=window_size,
            channels_to_replace=channels_to_replace,
            channels_to_use=channels_to_use,
        )
        signal_return = signal_data_filtered

        # Save the signal, scores and filtered signal of the first channel as pandas dataframes
        # df_signal = pd.DataFrame(signal_data[0])
        # df_signal.columns = ['signal']
        # df_signal['time'] = df_signal.index / sampling_frequency
        # df_signal['score'] = df_signal['time'].apply(lambda x: scores.get(x, np.nan))
        # df_signal.to_csv(f'./data/01_intermediate_analysis/{record_name.split("/")[-1]}.csv', index=False)
    else:
        signal_return = signal_data

    return signal_return, signal_channels, sampling_frequency


def get_max_key(score: Dict):
    """
    Returns the key with the maximum value in a dictionary.
    """

    # Get the maximum value
    max_value = max(score.values())

    # Get a list of keys with max value
    max_keys = [key for key, value in score.items() if value == max_value]

    # Calculate the middle index
    mid = len(max_keys) // 2

    # If the list has an even number of elements
    if len(max_keys) % 2 == 0:
        # Return a random key from the two middle ones
        return random.choice(max_keys[mid - 1 : mid + 1])
    else:
        # Otherwise, return the key at the middle index
        return max_keys[mid]


def train_torch_model(
    model, train_loader, val_loader, device, params, model_folder, checkpoint_path
):
    """
    Train a PyTorch model for a specified number of epochs, saving the model with the lowest validation loss.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        device (torch.device): The device (CPU or GPU) where the model and data are loaded.
        params (dict): A dictionary of hyperparameters for training, including 'learning_rate' and 'max_epochs'.
        model_folder (str): The directory where TensorBoard logs will be saved.
        checkpoint_path (str): The file path where the model state with the lowest validation loss will be saved.

    Returns:
        model (torch.nn.Module): The trained model. If saving checkpoints was enabled, this will return
                                the model state at the epoch with the lowest validation loss.
    """

    # Adjust path
    model_folder = model_folder.lower()

    # Set up criterion and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Set up scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, verbose=True
    )

    # Set up TensorBoard writer
    writer = SummaryWriter(model_folder)

    # Move model to the device
    model = model.to(device)
    best_model_wts = model.state_dict().copy()

    # If a checkpoint is provided, load the state of the model
    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.lower()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {start_epoch}.")
        if start_epoch >= params["max_epochs"]:
            print(
                f"Checkpoint epoch {start_epoch} is greater than max epochs {params['max_epochs']}. Exiting."
            )
            return model

    # Training loop
    best_auc = 0.
    min_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(start_epoch, params["max_epochs"]):
        print(
            f'Starting epoch {epoch}/{params["max_epochs"]} with {len(train_loader)} batches'
        )

        model.train()
        train_loss = 0.0
        train_auc = []
        for batch in tqdm(train_loader):
            inputs = batch["image"]
            features = batch["features"]
            labels = batch["label"]

            inputs = inputs.to(device)
            labels = labels.to(device)
            features = features.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, features)
            loss = criterion(outputs.view(-1), labels.float())

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            labels_aux = labels.clone()
            outputs_aux = outputs.clone()
            labels_np = labels_aux.detach().cpu().numpy()
            if len(np.unique(labels_np)) > 1:
                batch_auc = roc_auc_score(labels_np, outputs_aux.detach().cpu().numpy())
            else:
                batch_auc = np.nan
            train_auc.append(batch_auc)

            writer.add_scalar("train_loss", loss.item(), epoch)

        # Evaluate on validation data
        print(
            f'Validating epoch {epoch}/{params["max_epochs"]} with {len(val_loader)} batches'
        )
        model.eval()
        val_loss = 0.0
        val_auc = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs = batch["image"]
                features = batch["features"]
                labels = batch["label"]

                inputs = inputs.to(device)
                labels = labels.to(device)
                features = features.to(device)

                outputs = model(inputs, features)
                loss = criterion(outputs.view(-1), labels.float())

                labels_np = labels.cpu().numpy()
                if len(np.unique(labels_np)) > 1:
                    batch_auc = roc_auc_score(labels_np, outputs.cpu().numpy())
                else:
                    batch_auc = np.nan
                val_auc.append(batch_auc)

                val_loss += loss.item()

        # Calculate and store average loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_auc = np.nanmean(val_auc)
        avg_train_auc = np.nanmean(train_auc)
        writer.add_scalar("val_loss", avg_val_loss, epoch)
        print(
            f"Epoch {epoch}, avg. train loss: {round(avg_train_loss,4)}, avg. train AUC: {round(avg_train_auc,4)}, avg. val loss: {round(avg_val_loss,4)}, avg. val AUC: {round(avg_val_auc,4)}.\n Best so far: Epoch {best_epoch} with avg. val loss {round(min_val_loss,4)} and avg. val AUC {round(best_auc,4)}.\n Lr was {optimizer.param_groups[0]['lr']}."
        )

        # Adjust learning rate
        scheduler.step(avg_val_loss)

        # Save model at the end of every epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{model_folder}/checkpoint_last.pth")

        # If current model has the lowest validation loss so far, save it as the best model
        if avg_val_loss < min_val_loss:
            torch.save(model.state_dict(), f"{model_folder}/checkpoint_best.pth")
            min_val_loss = avg_val_loss
            best_auc = avg_val_auc
            best_epoch = epoch
            best_model_wts = model.state_dict().copy()
            torch.save(checkpoint, f"{model_folder}/checkpoint_best.pth")

    writer.close()

    if USE_BEST_MODEL:
        print(
            f"Going forward, using best model from epoch {best_epoch} with avg. val loss {round(min_val_loss,4)} and avg. val AUC {round(best_auc,4)}."
        )
        model.load_state_dict(best_model_wts)
    else:
        print(
            f"Going forward, using last model from epoch {epoch} with avg. val loss {round(avg_val_loss,4)} and avg. val AUC {round(avg_val_auc,4)}."
        )

    return model
