#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
from helper_code import *
import librosa
import json
import pandas as pd
import numpy as np, os, sys
import mne
import random
import re
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import joblib
import timm
import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from typing import Dict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


################################################################################
#
# Parameters
#
################################################################################
# Device settings
PARAMS_DEVICE = {"num_workers": 10} #os.cpu_count()}
print(f"CPU count: {os.cpu_count()}")
print(PARAMS_DEVICE)
USE_GPU = True

# Torch usage
USE_TORCH = False

# Recordings to use
NUM_HOURS_TO_USE = -1 # This currently uses the recording files, not hours
SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING = 120
FILTER_SIGNALS = False

# EEG usage
EEG_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz'] # ['F3', 'P3', 'F4', 'P4'] # ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'Fpz', 'Oz', 'F9']
BIPOLAR_MONTAGES = None # [('F3','P3'), ('F4','P4')]
NUM_HOURS_EEG = NUM_HOURS_TO_USE

# ECG usage
USE_ECG = False
ECG_CHANNELS = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2'] # ECG, ECG1, ECG2, ECGL, ECGR
NUM_HOURS_ECG = NUM_HOURS_TO_USE

# OTHER usage
USE_OTHER = False
OTHER_CHANNELS = ['SpO2', 'EMG1', 'EMG2', 'EMG3', 'LAT1', 'LAT2', 'LOC', 'ROC', 'LEG1', 'LEG2']
NUM_HOURS_OTHER = NUM_HOURS_TO_USE

# REF usage
USE_REF = False
REF_CHANNELS = ['RAT1', 'RAT2', 'REF', 'C2', 'A1', 'A2', 'BIP1', 'BIP2', 'BIP3', 'BIP4', 'Cb2', 'M1', 'M2', 'In1-Ref2', 'In1-Ref3']
NUM_HOURS_REF = NUM_HOURS_TO_USE

# Imputation
IMPUTE = True
IMPUTE_METHOD = 'constant' # 'mean', 'median', 'most_frequent', 'constant'
IMPUTE_CONSTANT_VALUE = -1

# Model and training paramters
PARAMS_TORCH = {'batch_size': 16, 'val_size': 0.3, 'max_epochs': 1, 'pretrained': True, 'devices': 1, 'num_nodes': 1}
C_MODEL = "rf" # "xgb" or "rf
PARAMS_RF = {'n_estimators': 100, 'max_depth': 8, 'max_leaf_nodes': None, 'random_state': 42, 'n_jobs': 8}
PARAMS_XGB = {'max_depth': 8, 'eval_metric': 'auc', 'nthread': 8}


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Save all parameters as json file
    params = {"PARAMS_DEVICE": PARAMS_DEVICE, "NUM_HOURS_TO_USE": NUM_HOURS_TO_USE, "SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING": SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING, "EEG_CHANNELS": EEG_CHANNELS, "BIPOLAR_MONTAGES": BIPOLAR_MONTAGES, "NUM_HOURS_EEG": NUM_HOURS_EEG, "USE_ECG": USE_ECG, "ECG_CHANNELS": ECG_CHANNELS, "NUM_HOURS_ECG": NUM_HOURS_ECG, "USE_OTHER": USE_OTHER, "OTHER_CHANNELS": OTHER_CHANNELS, "NUM_HOURS_OTHER": NUM_HOURS_OTHER, "USE_REF": USE_REF, "REF_CHANNELS": REF_CHANNELS, "NUM_HOURS_REF": NUM_HOURS_REF, "USE_TORCH": USE_TORCH, "IMPUTE": IMPUTE, "IMPUTE_METHOD": IMPUTE_METHOD, "IMPUTE_CONSTANT_VALUE": IMPUTE_CONSTANT_VALUE, "PARAMS_TORCH": PARAMS_TORCH, "C_MODEL": C_MODEL, "PARAMS_RF": PARAMS_RF, "PARAMS_XGB": PARAMS_XGB}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    with open(os.path.join(model_folder, "params.json"), "w") as f:
        json.dump(params, f)

    # Parameters
    params_torch = PARAMS_TORCH
    c_model = C_MODEL
    params_rf = PARAMS_RF
    params_xgb = PARAMS_XGB

    if USE_TORCH:
        # Get device
        if USE_GPU:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = "cpu"
            accelerator = "cpu"
        print(f"Using device {device} and accelerator {accelerator}")
        if c_model=="rf":
            print(f"Train with torch. Torch params: {params_torch}, {c_model} params: {params_rf}")
        elif c_model=="xgb":
            print(f"Train with torch. Torch params: {params_torch}, {c_model} params: {params_xgb}")
        else:
            raise ValueError(f"No such c_model: {c_model}")
    else:
        if c_model=="rf":
            print(f"Train without torch. {c_model} params: {params_rf}")
        elif c_model=="xgb":
            print(f"Train without torch. {c_model} params: {params_xgb}")
        else:
            raise ValueError(f"No such c_model: {c_model}")

    # Find data files.
    start_time = time.time()
    if verbose >= 1:
        print(f'Finding the challenge data in {data_folder}...')
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)
    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    torch_model_eeg = None
    torch_model_ecg = None
    torch_model_other = None
    torch_model_ref = None
    if USE_TORCH:
        # Split into train and validation set
        num_val = int(num_patients * params_torch['val_size'])
        num_train = num_patients - num_val
        patient_ids_aux = patient_ids.copy()
        random.Random(42).shuffle(patient_ids_aux) #TODO: Is this good?
        train_ids = patient_ids_aux[:num_train]
        val_ids = patient_ids_aux[num_train:]

        # Get EEG DL data
        train_dataset_eeg = RecordingsDataset(data_folder, patient_ids = train_ids, device=device, group = "EEG", hours_to_use=None)
        val_dataset_eeg = RecordingsDataset(data_folder, patient_ids = val_ids, device=device, group = "EEG", hours_to_use=None)
        torch_dataset_eeg = RecordingsDataset(data_folder, patient_ids = patient_ids, device=device, group = "EEG", hours_to_use=NUM_HOURS_EEG)
        train_loader_eeg = DataLoader(train_dataset_eeg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=True, pin_memory=True)
        val_loader_eeg = DataLoader(val_dataset_eeg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)
        data_loader_eeg = DataLoader(torch_dataset_eeg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)

        # Find last checkpoint
        model_folder_eeg = os.path.join(model_folder, "EEG")
        checkpoint_path_eeg = get_last_chkpt(model_folder_eeg)

        # Define torch model
        torch_model_eeg = get_tv_model(batch_size=params_torch['batch_size'], d_size=len(train_loader_eeg), pretrained=params_torch['pretrained'], channel_size=len(EEG_CHANNELS))
        trainer_eeg = pl.Trainer(
            accelerator=accelerator,
            #devices=params_torch["devices"],
            #num_nodes=params_torch["num_nodes"],
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_last=True, save_top_k=1)],
            log_every_n_steps=1,
            max_epochs=params_torch['max_epochs'],
            enable_progress_bar=True,
            logger=TensorBoardLogger(model_folder_eeg, name=''),
        )
        trainer_eeg.logger._default_hp_metric = False

        # Train EEG torch model
        print("Start training EEG torch model...")
        start_time_torch_eeg = time.time()
        trainer_eeg.fit(torch_model_eeg, train_loader_eeg, val_loader_eeg, ckpt_path=checkpoint_path_eeg)
        print(f"Finished training EEG torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch_eeg)/60,4)} min.")

        # Get EEG predictions
        print("Start predicting EEG torch features ...")
        output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg = torch_prediction(torch_model_eeg, data_loader_eeg, device)
        print("Done with EEG torch.")

        if USE_ECG:
            # Get ECG DL data
            train_dataset_ecg = RecordingsDataset(data_folder, patient_ids = train_ids, device=device, group = "ECG", hours_to_use=None)
            val_dataset_ecg = RecordingsDataset(data_folder, patient_ids = val_ids, device=device, group = "ECG", hours_to_use=None)
            torch_dataset_ecg = RecordingsDataset(data_folder, patient_ids = patient_ids, device=device, group = "ECG", hours_to_use=NUM_HOURS_ECG)
            train_loader_ecg = DataLoader(train_dataset_ecg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=True, pin_memory=True)
            val_loader_ecg = DataLoader(val_dataset_ecg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)
            data_loader_ecg = DataLoader(torch_dataset_ecg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)

            # Find last checkpoint
            model_folder_ecg = os.path.join(model_folder, "ECG")
            checkpoint_path_ecg = get_last_chkpt(model_folder_ecg)

            # Define torch model
            torch_model_ecg = get_tv_model(batch_size=params_torch['batch_size'], d_size=len(train_loader_ecg), pretrained=params_torch['pretrained'], channel_size=len(ECG_CHANNELS))
        
            trainer_ecg = pl.Trainer(
                accelerator=accelerator,
                devices=params_torch["devices"],
                num_nodes=params_torch["num_nodes"],
                callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_last=True, save_top_k=1)],
                log_every_n_steps=1,
                max_epochs=params_torch['max_epochs'],
                enable_progress_bar=True,
                logger=TensorBoardLogger(model_folder_ecg, name=''),
            )
            trainer_ecg.logger._default_hp_metric = False

            # Train ECG torch model
            print("Start training ECG torch model...")
            start_time_torch = time.time()
            trainer_ecg.fit(torch_model_ecg, train_loader_ecg, val_loader_ecg, ckpt_path=checkpoint_path_ecg)
            print(f"Finished training ECG torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min.")

            # Get ECG predictions
            print("Start predicting ECG torch features ...")
            output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg = torch_prediction(torch_model_ecg, data_loader_ecg, device)
            print("Done with ECG torch.")

        if USE_REF:
            # Get REF DL data
            train_dataset_ref = RecordingsDataset(data_folder, patient_ids = train_ids, device=device, group = "REF", hours_to_use=None)
            val_dataset_ref = RecordingsDataset(data_folder, patient_ids = val_ids, device=device, group = "REF", hours_to_use=None)
            torch_dataset_ref = RecordingsDataset(data_folder, patient_ids = patient_ids, device=device, group = "REF", hours_to_use=NUM_HOURS_REF)
            train_loader_ref = DataLoader(train_dataset_ref, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=True, pin_memory=True)
            val_loader_ref = DataLoader(val_dataset_ref, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)
            data_loader_ref = DataLoader(torch_dataset_ref, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)

            # Find last checkpoint
            model_folder_ref = os.path.join(model_folder, "REF")
            checkpoint_path_ref = get_last_chkpt(model_folder_ref)

            # Define torch model
            torch_model_ref = get_tv_model(batch_size=params_torch['batch_size'], d_size=len(train_loader_ref), pretrained=params_torch['pretrained'], channel_size=len(REF_CHANNELS))
        
            trainer_ref = pl.Trainer(
                accelerator=accelerator,
                devices=params_torch["devices"],
                num_nodes=params_torch["num_nodes"],
                callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_last=True, save_top_k=1)],
                log_every_n_steps=1,
                max_epochs=params_torch['max_epochs'],
                enable_progress_bar=True,
                logger=TensorBoardLogger(model_folder_ref, name=''),
            )
            trainer_ref.logger._default_hp_metric = False

            # Train REF torch model
            print("Start training REF torch model...")
            start_time_torch = time.time()
            trainer_ref.fit(torch_model_ref, train_loader_ref, val_loader_ref, ckpt_path=checkpoint_path_ref)
            print(f"Finished training REF torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min.")

            # Get REF predictions
            print("Start predicting REF torch features ...")
            output_list_ref, patient_id_list_ref, hour_list_ref, quality_list_ref = torch_prediction(torch_model_ref, data_loader_ref, device)
            print("Done with REF torch.")

        if USE_OTHER:
            # Get OTHER DL data
            train_dataset_other = RecordingsDataset(data_folder, patient_ids = train_ids, device=device, group = "OTHER", hours_to_use=None)
            val_dataset_other = RecordingsDataset(data_folder, patient_ids = val_ids, device=device, group = "OTHER", hours_to_use=None)
            torch_dataset_other = RecordingsDataset(data_folder, patient_ids = patient_ids, device=device, group = "OTHER", hours_to_use=NUM_HOURS_OTHER)
            train_loader_other = DataLoader(train_dataset_other, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=True, pin_memory=True)
            val_loader_other = DataLoader(val_dataset_other, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)
            data_loader_other = DataLoader(torch_dataset_other, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)

            # Find last checkpoint
            model_folder_other = os.path.join(model_folder, "OTHER")
            checkpoint_path_other = get_last_chkpt(model_folder_other)

            # Define torch model
            torch_model_other = get_tv_model(batch_size=params_torch['batch_size'], d_size=len(train_loader_other), pretrained=params_torch['pretrained'], channel_size=len(OTHER_CHANNELS))
        
            trainer_other = pl.Trainer(
                accelerator=accelerator,
                devices=params_torch["devices"],
                num_nodes=params_torch["num_nodes"],
                callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_last=True, save_top_k=1)],
                log_every_n_steps=1,
                max_epochs=params_torch['max_epochs'],
                enable_progress_bar=True,
                logger=TensorBoardLogger(model_folder_other, name=''),
            )
            trainer_other.logger._default_hp_metric = False

            # Train OTHER torch model
            print("Start training OTHER torch model...")
            start_time_torch = time.time()
            trainer_other.fit(torch_model_other, train_loader_other, val_loader_other, ckpt_path=checkpoint_path_other)
            print(f"Finished training OTHER torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min.")

            # Get OTHER predictions
            print("Start predicting OTHER torch features ...")
            output_list_other, patient_id_list_other, hour_list_other, quality_list_other = torch_prediction(torch_model_other, data_loader_other, device)
            print("Done with OTHER torch.")

        print("Done with torch")
        
    print("Calculating features...")
    features = list()
    feature_names = list()
    outcomes = list()
    patients = list()
    hospitals = list()
    recording_meta_infos = list()
    cpcs = list()
    patient_ids_aux = list()
    outcome_probabilities_torch_eeg_aux = list()
    outcome_probabilities_torch_ecg_aux = list()
    outcome_probabilities_torch_ref_aux = list()
    outcome_probabilities_torch_other_aux = list()
    outcome_flags_torch_eeg_aux = list()
    outcome_flags_torch_ecg_aux = list()
    outcome_flags_torch_ref_aux = list()
    outcome_flags_torch_other_aux = list()
    for i in tqdm(range(num_patients)):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data and extract features
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_features, current_feature_names, hospital, recording_infos = get_features(data_folder, patient_ids[i])

        if USE_TORCH:
            # Get torch predictions
            agg_outcome_probabilities_torch_eeg, outcome_probabilities_torch_eeg, outcome_flags_torch_eeg = torch_predictions_for_patient(output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg, patient_ids[i])
            current_features = np.hstack((current_features, outcome_probabilities_torch_eeg))
            current_feature_names = np.hstack((current_feature_names, [f"prob_eeg_torch_{i}" for i in range(len(outcome_probabilities_torch_eeg))]))
            outcome_probabilities_torch_eeg_aux.append(outcome_probabilities_torch_eeg)
            outcome_flags_torch_eeg_aux.append(outcome_flags_torch_eeg)
            if USE_ECG:
                agg_outcome_probabilities_torch_ecg, outcome_probabilities_torch_ecg, outcome_flags_torch_ecg = torch_predictions_for_patient(output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg, patient_ids[i])
                current_features = np.hstack((current_features, outcome_probabilities_torch_ecg))
                current_feature_names = np.hstack((current_feature_names, [f"prob_ecg_torch_{i}" for i in range(len(outcome_probabilities_torch_ecg))]))
                outcome_probabilities_torch_ecg_aux.append(outcome_probabilities_torch_ecg)
                outcome_flags_torch_ecg_aux.append(outcome_flags_torch_ecg)
            if USE_REF:
                agg_outcome_probabilities_torch_ref, outcome_probabilities_torch_ref, outcome_flags_torch_ref = torch_predictions_for_patient(output_list_ref, patient_id_list_ref, hour_list_ref, quality_list_ref, patient_ids[i])
                current_features = np.hstack((current_features, outcome_probabilities_torch_ref))
                current_feature_names = np.hstack((current_feature_names, [f"prob_ref_torch_{i}" for i in range(len(outcome_probabilities_torch_ref))]))
                outcome_probabilities_torch_ref_aux.append(outcome_probabilities_torch_ref)
                outcome_flags_torch_ref_aux.append(outcome_flags_torch_ref)
            if USE_OTHER:
                agg_outcome_probabilities_torch_other, outcome_probabilities_torch_other, outcome_flags_torch_other = torch_predictions_for_patient(output_list_other, patient_id_list_other, hour_list_other, quality_list_other, patient_ids[i])
                current_features = np.hstack((current_features, outcome_probabilities_torch_other))
                current_feature_names = np.hstack((current_feature_names, [f"prob_other_torch_{i}" for i in range(len(outcome_probabilities_torch_other))]))
                outcome_probabilities_torch_other_aux.append(outcome_probabilities_torch_other)
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
    train_pd['patient_ids'] = patients
    train_pd['outcome'] = outcomes
    train_pd['cpc'] = cpcs
    train_pd['hospital'] = hospitals
    first_column = train_pd.pop('patient_ids')
    train_pd.insert(0, 'patient_ids', first_column)
    for key in recording_meta_infos[0][0].keys():
        values_aux = list()
        for i in range(len(recording_meta_infos)):
            values_aux.append(recording_meta_infos[i][0][key])
        train_pd[key] = values_aux
    train_pd.to_csv(os.path.join(model_folder, 'train_features.csv'), index=False)

    # Impute any missing features; use the mean value by default.
    if IMPUTE:
        print("Imputing features...")
        imputer = SimpleImputer(strategy=IMPUTE_METHOD, fill_value=IMPUTE_CONSTANT_VALUE).fit(features)
        features = imputer.transform(features)
    else:
        imputer = None

    # Save the imputed features.
    print("Saving imputed features...")
    train_imputed_pd = pd.DataFrame(features, columns=feature_names[0])
    train_imputed_pd['patient_ids'] = patients
    train_imputed_pd['outcome'] = outcomes
    train_imputed_pd['cpc'] = cpcs
    train_imputed_pd['hospital'] = hospitals
    for key in recording_meta_infos[0][0].keys():
        values_aux = list()
        for i in range(len(recording_meta_infos)):
            values_aux.append(recording_meta_infos[i][0][key])
        train_imputed_pd[key] = values_aux
    train_imputed_pd.to_csv(os.path.join(model_folder, 'train_imputed_features.csv'), index=False)

    # Train the models.
    print("Start training challenge models...")
    if c_model == "rf":
        outcome_model = RandomForestClassifier(**params_rf)
        cpc_model = RandomForestRegressor(**params_rf)
    elif c_model == "xgb":
        outcome_model = xgb.XGBClassifier(**params_xgb)
        cpc_model = xgb.XGBRegressor(**params_xgb)
    outcome_model.fit(features, outcomes.ravel())
    cpc_model.fit(features, cpcs.ravel())
    
    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model, torch_model_eeg=torch_model_eeg, torch_model_ecg=torch_model_ecg, torch_model_ref=torch_model_ref, torch_model_other=torch_model_other)

    # Plot and save feature importance.
    feature_importance = outcome_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(12, min(len(feature_importance)/2,100)))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names[0])[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "feature_importance.png"))
    plt.close()

    if verbose >= 1:
        print(f'Done after {round((time.time()-start_time)/60,4)} min.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    model = joblib.load(filename)
    file_path_eeg = os.path.join(model_folder, 'eeg', 'checkpoint.pth')
    file_path_ecg = os.path.join(model_folder, 'ecg', 'checkpoint.pth')
    file_path_ref = os.path.join(model_folder, 'ref', 'checkpoint.pth')
    file_path_other = os.path.join(model_folder, 'other', 'checkpoint.pth')
    if USE_TORCH:
        model["torch_model_eeg"] = load_last_pt_ckpt(file_path_eeg, channel_size=len(EEG_CHANNELS))
        if USE_ECG:
            model["torch_model_ecg"] = load_last_pt_ckpt(file_path_ecg, channel_size=len(ECG_CHANNELS))
        else:
            model["torch_model_ecg"] = None
        if USE_REF:
            model["torch_model_ref"] = load_last_pt_ckpt(file_path_ref, channel_size=len(REF_CHANNELS))
        else:
            model["torch_model_ref"] = None
        if USE_OTHER:
            model["torch_model_other"] = load_last_pt_ckpt(file_path_other, channel_size=len(OTHER_CHANNELS))
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
def run_challenge_models(models, data_folder, patient_id, verbose):

    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    torch_model_eeg = models['torch_model_eeg']
    torch_model_ecg = models['torch_model_ecg']
    torch_model_ref = models['torch_model_ref']
    torch_model_other = models['torch_model_other']

    # Load data.
    features, _, _, _ = get_features(data_folder, patient_id) 

    # Torch prediction
    if USE_TORCH:
        if USE_GPU:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = "cpu"
        data_set_eeg = RecordingsDataset(data_folder, patient_ids = [patient_id], device = device, load_labels = False, group="EEG", hours_to_use = NUM_HOURS_EEG)
        data_loader_eeg = DataLoader(data_set_eeg, batch_size=1, num_workers=PARAMS_DEVICE["num_workers"], shuffle=False)
        output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg = torch_prediction(torch_model_eeg, data_loader_eeg, device)
        agg_outcome_probability_torch_eeg, outcome_probabilities_torch_eeg, outcome_flags_torch_eeg = torch_predictions_for_patient(output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg, patient_id)
        features = np.hstack((features, outcome_probabilities_torch_eeg))
        if USE_ECG:
            data_set_ecg = RecordingsDataset(data_folder, patient_ids = [patient_id], device = device, load_labels = False, group="ECG", hours_to_use = NUM_HOURS_ECG)
            data_loader_ecg = DataLoader(data_set_ecg, batch_size=1, num_workers=PARAMS_DEVICE["num_workers"], shuffle=False)
            output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg = torch_prediction(torch_model_ecg, data_loader_ecg, device)
            agg_outcome_probability_torch_ecg, outcome_probabilities_torch_ecg, outcome_flags_torch_ecg = torch_predictions_for_patient(output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg, patient_id)
            features = np.hstack((features, outcome_probabilities_torch_ecg))
        if USE_REF:
            data_set_ref = RecordingsDataset(data_folder, patient_ids = [patient_id], device = device, load_labels = False, group="REF", hours_to_use = NUM_HOURS_REF)
            data_loader_ref = DataLoader(data_set_ref, batch_size=1, num_workers=PARAMS_DEVICE["num_workers"], shuffle=False)
            output_list_ref, patient_id_list_ref, hour_list_ref, quality_list_ref = torch_prediction(torch_model_ref, data_loader_ref, device)
            agg_outcome_probability_torch_ref, outcome_probabilities_torch_ref, outcome_flags_torch_ref = torch_predictions_for_patient(output_list_ref, patient_id_list_ref, hour_list_ref, quality_list_ref, patient_id)
            features = np.hstack((features, outcome_probabilities_torch_ref))
        if USE_OTHER:
            data_set_other = RecordingsDataset(data_folder, patient_ids = [patient_id], device = device, load_labels = False, group="OTHER", hours_to_use = NUM_HOURS_OTHER)
            data_loader_other = DataLoader(data_set_other, batch_size=1, num_workers=PARAMS_DEVICE["num_workers"], shuffle=False)
            output_list_other, patient_id_list_other, hour_list_other, quality_list_other = torch_prediction(torch_model_other, data_loader_other, device)
            agg_outcome_probability_torch_other, outcome_probabilities_torch_other, outcome_flags_torch_other = torch_predictions_for_patient(output_list_other, patient_id_list_other, hour_list_other, quality_list_other, patient_id)
            features = np.hstack((features, outcome_probabilities_torch_other))

    # Impute missing data.
    features = features.reshape(1, -1)
    if imputer is not None:
        features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    #outcome_probability = agg_outcome_probability_torch
    #outcome = 1 if outcome_probability > 0.5 else 0
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

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


def find_recording_files(data_folder, patient_id, group=""):
    record_names = list()
    patient_folder = os.path.join(data_folder, patient_id)
    for file_name in sorted(os.listdir(patient_folder)):
        if not file_name.startswith('.') and file_name.endswith(f'{group}.hea'):
            root, ext = os.path.splitext(file_name)
            record_name = '_'.join(root.split('_')[:-1])
            record_names.append(record_name)
    return sorted(record_names)


def get_last_chkpt(model_folder):
    if os.path.exists(model_folder):
        # Find last version folder
        last_version = 0
        for folder in os.listdir(model_folder):
            if "version_" in folder:
                if "checkpoints" in os.path.join(model_folder, folder):
                    last_version = max(last_version, int(folder.split("_")[-1]))
        last_version = f"version_{last_version}"
        if os.path.isfile(f"{model_folder}/{last_version}/checkpoints/last.ckpt"):
            checkpoint_path = f"{model_folder}/{last_version}/checkpoints/last.ckpt"
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None

    if checkpoint_path is not None:
        print("Resuming from checkpoint: ", checkpoint_path)
    else:
        print("No checkpoint found. Starting from scratch.")
    
    return checkpoint_path


def torch_prediction(model, data_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        output_list = []
        patient_id_list = []
        hour_list = []
        quality_list = []
        for _, batch in enumerate(tqdm(data_loader)):
            data, targets, ids, hours, qualities = batch["image"], batch["label"], batch["id"], batch["hour"], batch["quality"]
            data = data.to(device)
            outputs = model(data)
            outputs = torch.sigmoid(outputs)
            output_list = output_list + outputs.cpu().numpy().tolist()
            patient_id_list = patient_id_list + ids
            hour_list = hour_list + list(hours.cpu().detach().numpy())
            quality_list = quality_list + list(qualities.cpu().detach().numpy())
    return output_list, patient_id_list, hour_list, quality_list


def torch_predictions_for_patient(output_list, patient_id_list, hour_list, quality_list, patient_id, max_hours=72, min_quality=0, num_signals=None):
    # Get the predictions for the patient
    patient_mask = np.array([True if p == patient_id else False for p in patient_id_list])
    if len(patient_mask) == 0:
        outcome_probabilities_torch = np.array([])
        hours_patients = np.array([])
    else:
        outcome_probabilities_torch = np.array(output_list)[patient_mask]
        hours_patients = np.array(hour_list)[patient_mask].astype(int).tolist()

        if len(outcome_probabilities_torch[0]) == 1:
            outcome_probabilities_torch = [i[0] for i in outcome_probabilities_torch]
        else:
            raise ValueError("The torch model should only predict one value per patient.")

    #  Get values
    outcome_probabilities_torch = [outcome_probabilities_torch[hours_patients.index(hour)] if hour in hours_patients else np.nan for hour in range(max_hours)]
    if IMPUTE:
        outcome_probabilities_torch_imputed = pd.Series(outcome_probabilities_torch, dtype=object).bfill().tolist()
    else:
        outcome_probabilities_torch_imputed = pd.Series(outcome_probabilities_torch, dtype=object).tolist()
    outcome_flags_torch = [1 if hour in hours_patients else 0 for hour in range(max_hours)]

    # Aggregate the probabilities
    weights = range(max_hours)
    agg_outcome_probability_torch = [p * w for p, w in zip(outcome_probabilities_torch, weights) if not np.isnan(p)]
    weight_sum = sum([w for p, w in zip(outcome_probabilities_torch, weights) if not np.isnan(p)])
    if weight_sum == 0:
        agg_outcome_probability_torch = 0
    else:
        agg_outcome_probability_torch = sum(agg_outcome_probability_torch) / weight_sum

    # Find the median of the signals where outcome_flags_torch is 1 and only keep the num_signals around the median
    if num_signals is not None:
        if num_signals > max_hours:
            raise ValueError("num_signals should be smaller than max_hours.")
        median = np.median(np.where(np.array(outcome_flags_torch) == 1)[0])
        lower_bound = int(median - num_signals/2)
        upper_bound = int(median + num_signals/2)
        if lower_bound < 0:
            upper_bound = upper_bound + abs(lower_bound)
            lower_bound = 0
        if upper_bound > max_hours:
            lower_bound = lower_bound - (upper_bound - max_hours)
            upper_bound = max_hours
        outcome_probabilities_torch_imputed = outcome_probabilities_torch_imputed[lower_bound:upper_bound]
        outcome_flags_torch = outcome_flags_torch[lower_bound:upper_bound]

    return agg_outcome_probability_torch, outcome_probabilities_torch_imputed, outcome_flags_torch


def get_tv_model(model_name="densenet121", num_classes=1, batch_size=64, d_size=500, pretrained=False, channel_size=3):
    model = torchvisionModel(
            model_name=model_name,
            num_classes=num_classes,
            print_freq=250,
            batch_size=batch_size,
            d_size=d_size,
            pretrained=pretrained,
            channel_size=channel_size,
        )
    return model


# Load last checkpoint
def load_last_pt_ckpt(ckpt_path, channel_size):
    model = get_tv_model(channel_size=channel_size)
    if os.path.isfile(ckpt_path):
        print(
            f"Loading checkpoint from {ckpt_path}"
        )
        if "pth" in ckpt_path:
            checkpoint = torch.load(ckpt_path)
            state_dic = checkpoint["model"]
            model.load_state_dict(state_dic)
        elif "ckpt" in ckpt_path:
            model = model.load_from_checkpoint(ckpt_path)
        return model
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model, **torch_models):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)
    for name, torch_model in torch_models.items():
        if torch_model is not None:
            torch_model_folder = os.path.join(model_folder, name.split('_')[-1])
            if not os.path.exists(torch_model_folder):
                os.makedirs(torch_model_folder)
            file_path = os.path.join(torch_model_folder, 'checkpoint.pth')
            torch.save({"model": torch_model.state_dict()}, file_path)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.5, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=1, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=1, verbose='error')

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

    # Remove the first and last x seconds of the recording to avoid edge effects.
    num_samples_to_remove = int(SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING * resampling_frequency)
    if num_samples_to_remove > 0:
        if data.shape[1] > (2 * num_samples_to_remove + resampling_frequency * 60 * 5):
            data = data[:, num_samples_to_remove:-num_samples_to_remove]

    return data, resampling_frequency

# Load recording data.
def get_recording_features(recording_ids, recording_id_to_use, data_folder, patient_id, group, channels_to_use):
    if group == "EEG":
        if BIPOLAR_MONTAGES is not None:
            channel_length = len(BIPOLAR_MONTAGES)
        else:
            channel_length = len(channels_to_use)
        dummy_channels = channel_length * 4
        recording_feature_names = np.array((np.array([f"delta_psd_mean_c_{i}_hour_{recording_id_to_use}" for i in range(channel_length)]), np.array([f"theta_psd_mean_c_{i}_hour_{recording_id_to_use}" for i in range(channel_length)]), np.array([f"alpha_psd_mean_c_{i}_hour_{recording_id_to_use}" for i in range(channel_length)]), np.array([f"beta_psd_mean_c_{i}_hour_{recording_id_to_use}" for i in range(channel_length)]))).T.flatten()
    elif (group == "ECG") or (group == "REF") or (group == "OTHER"):
        channel_length = len(channels_to_use)
        dummy_channels = channel_length * 2
        recording_feature_names = np.array((np.array([f"{group}_mean_c_{i}_hour_{recording_id_to_use}" for i in range(channel_length)]), np.array([f"{group}_std_c_{i}_hour_{recording_id_to_use}" for i in range(channel_length)]))).T.flatten()
    else:
        raise ValueError("Group should be either EEG, ECG, REF or OTHER")
    
    if len(recording_ids) > 0:
        if abs(recording_id_to_use) <= len(recording_ids):
            recording_id = recording_ids[recording_id_to_use]
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data_wrapper(recording_location)
                hea_file = load_text_file(recording_location + '.hea')
                utility_frequency = get_utility_frequency(hea_file)
                quality = get_quality(hea_file)
                hour = get_hour(hea_file)
                if all(channel in channels for channel in channels_to_use) or group != "EEG":
                    data, channels = reduce_channels(data, channels, channels_to_use)
                    data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                    if group == "EEG":
                        if BIPOLAR_MONTAGES is not None:
                            data = np.array([data[channels.index(montage[0]), :] - data[channels.index(montage[1]), :] for montage in BIPOLAR_MONTAGES])
                        recording_features = get_eeg_features(data, sampling_frequency)
                    elif (group == "ECG") or (group == "REF") or (group == "OTHER"):
                        features = get_ecg_features(data)
                        recording_features = expand_channels(features, channels, channels_to_use).flatten()
                    else:
                        raise NotImplementedError(f"Group {group} not implemented.")
                else:
                    print(f"For patient {patient_id} recording {recording_id} the channels {channels_to_use} are not all available. Only {channels} are available.")
                    recording_features = float('nan') * np.ones(dummy_channels) # 2 bipolar channels * 4 features / channel
            else:
                recording_features = float('nan') * np.ones(dummy_channels) # 2 bipolar channels * 4 features / channel
                sampling_frequency = utility_frequency = channels = quality = hour = None
        else:
            recording_features = float('nan') * np.ones(dummy_channels) # 2 bipolar channels * 4 features / channel
            sampling_frequency = utility_frequency = channels = recording_id = quality = hour = None
    else:
        recording_features = float('nan') * np.ones(dummy_channels) # 2 bipolar channels * 4 features / channel
        sampling_frequency = utility_frequency = channels = recording_id = quality = hour = None

    return recording_features, recording_feature_names, sampling_frequency, utility_frequency, channels, recording_id, quality, hour

# Extract features from the data.
def get_features(data_folder, patient_id, return_as_dict=False):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids_eeg = find_recording_files(data_folder, patient_id, "EEG")
    use_last_hours_eeg, hours_eeg, start_eeg = get_correct_hours(NUM_HOURS_EEG)
    if USE_ECG:
        recording_ids_ecg = find_recording_files(data_folder, patient_id, "ECG")
        use_last_hours_ecg, hours_ecg, start_ecg = get_correct_hours(NUM_HOURS_ECG)
    else:
        recording_ids_ecg = use_last_hours_ecg = hours_ecg = start_ecg = None
    if USE_REF:
        recording_ids_ref = find_recording_files(data_folder, patient_id, "REF")
        use_last_hours_ref, hours_ref, start_ref = get_correct_hours(NUM_HOURS_REF)
    else:
        recording_ids_ref = use_last_hours_ref = hours_ref = start_ref = None
    if USE_OTHER:
        recording_ids_other = find_recording_files(data_folder, patient_id, "OTHER")
        use_last_hours_other, hours_other, start_other = get_correct_hours(NUM_HOURS_OTHER)
    else:
        recording_ids_other = use_last_hours_other = hours_other = start_other = None

    # Extract patient features.
    patient_features, patient_feature_names = get_patient_features(patient_metadata, recording_ids_eeg)
    hospital = get_hospital(patient_metadata)

    # Extract recording features.
    feature_types = ['eeg', 'ecg', 'ref', 'other']
    use_flags = {'eeg': True, 'ecg': USE_ECG, 'ref': USE_REF, 'other': USE_OTHER}
    starts = {'eeg': start_eeg, 'ecg': start_ecg, 'ref': start_ref, 'other': start_other}
    hours = {'eeg': hours_eeg, 'ecg': hours_ecg, 'ref': hours_ref, 'other': hours_other}
    use_last_hours = {'eeg': use_last_hours_eeg, 'ecg': use_last_hours_ecg, 'ref': use_last_hours_ref, 'other': use_last_hours_other}
    recording_ids = {'eeg': recording_ids_eeg, 'ecg': recording_ids_ecg, 'ref': recording_ids_ref, 'other': recording_ids_other}
    channels_to_use = {'eeg': EEG_CHANNELS, 'ecg': ECG_CHANNELS, 'ref': REF_CHANNELS, 'other': OTHER_CHANNELS}
    recording_infos = {}
    feature_values = patient_features
    feature_names = patient_feature_names
    for feature_type in feature_types:
        if use_flags[feature_type]:
            feature_data = process_feature(feature_type, starts[feature_type], hours[feature_type], use_last_hours[feature_type], recording_ids[feature_type], channels_to_use[feature_type], data_folder, patient_id)
            recording_infos.update(feature_data)
            feature_values = np.hstack((feature_values, np.hstack(feature_data[f'{feature_type}_features'])))
            feature_names = np.hstack((feature_names, np.hstack(feature_data[f'{feature_type}_feature_names'])))
    
    if return_as_dict:
        return {k: v for k, v in zip(feature_names, feature_values)}
    else:
        return feature_values, feature_names, hospital, recording_infos

# Extract patient features from the data.
def get_patient_features(data, recording_ids_eeg):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    last_eeg_hour = np.max([int(r.split("_")[-1]) for r in recording_ids_eeg])

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm, last_eeg_hour))
    feature_names = ["age", "female", "male", "other", "rosc", "ohca", "shockable_rhythm", "ttm", "last_eeg_hour"]

    return features, feature_names

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=4.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T
    features = features.flatten()

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features


class RecordingsDataset(Dataset):
    def __init__(
        self, data_folder, patient_ids, device, group = "EEG", load_labels: bool=True, hours_to_use: int=None
    ):
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

        recording_locations_list = list()
        patient_ids_list = list()
        labels_list = list()
        for patient_id in patient_ids:
            patient_metadata = load_challenge_data(data_folder, patient_id)
            recording_ids = find_recording_files(data_folder, patient_id, self.group)
            if self.hours_to_use is not None:
                if abs(self.hours_to_use) < len(recording_ids):
                    if self.hours_to_use > 0:
                        recording_ids = recording_ids[:self.hours_to_use]
                    else:
                        recording_ids = recording_ids[self.hours_to_use:]
            #recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')
            #recording_metadata = load_text_file(recording_metadata_file)
            #hours = get_variable(recording_metadata, 'Hour', str)
            #qualities = get_variable(recording_metadata, 'Quality', str)
            if load_labels:
                current_outcome = get_outcome(patient_metadata)
            else:
                current_outcome = 0
            for recording_id in recording_ids:
                if not is_nan(recording_id):
                    recording_location_aux = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, self.group))
                    if os.path.exists(recording_location_aux + '.hea'):
                        recording_locations_list.append(recording_location_aux)
                        patient_ids_list.append(patient_id)
                        labels_list.append(current_outcome)
    
        self.recording_locations = recording_locations_list
        self.patient_ids = patient_ids_list
        self.labels = labels_list
        self.device = device
        self._precision = torch.float32

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load the data.
        try:
            signal_data, signal_channels, sampling_frequency = load_recording_data_wrapper(self.recording_locations[idx])
        except Exception as e:
            print("Error loading {}".format(self.recording_locations[idx]))
            raise e
        hea_file = load_text_file(self.recording_locations[idx] + '.hea')
        utility_frequency = get_utility_frequency(hea_file)
        signal_data, signal_channels = reduce_channels(signal_data, signal_channels, self.channels_to_use)

        # Preprocess the data. #TODO: Make this smarter. What to do if only a few seconds are available?
        sampling_frequency_old = sampling_frequency
        signal_data, sampling_frequency = preprocess_data(signal_data, sampling_frequency, utility_frequency)
        points_full_hour = 60*60*sampling_frequency
        available_points = signal_data.shape[1]/points_full_hour
        target_size = 901
        hop_length = max(int(round(signal_data.shape[1]/target_size,0)),1)
        spectrograms = librosa.feature.melspectrogram(y=signal_data, sr=sampling_frequency, n_mels=224, hop_length=hop_length)
        spectrograms = torch.from_numpy(spectrograms.astype(np.float32))
        spectrograms = nn.functional.normalize(spectrograms).to(self._precision)
        spectrograms = spectrograms.unsqueeze(0)
        spectrograms_resized = F.interpolate(spectrograms, size=(spectrograms.shape[2], target_size), mode='bilinear', align_corners=False)
        spectrograms = spectrograms_resized.squeeze(0)

        # Get the label.
        id = self.patient_ids[idx]
        label = self.labels[idx]
        hour = get_hour(hea_file)
        quality = get_quality(hea_file)
        label = torch.from_numpy(np.array(label).astype(np.float32)).to(self._precision)
        return_dict =  {"image": spectrograms.to(self._precision), "label": label.to(self._precision), "id": id, "hour": hour, "quality": quality}

        #print("------------------------------------------------------------------------")
        #print(f"Patiens: {id}, hour: {hour}, quality: {quality}, available_points: {available_points}, type: {self.group}, sampling_frequency_old: {sampling_frequency_old}, utility-freq: {utility_frequency}, sampling_frequency: {sampling_frequency}, hop_length: {hop_length}")
        #print(f"Signal shape: {signal_data.shape}")
        #print(f"Spectrograms shape: {spectrograms.shape}")

        return return_dict
    

class torchvisionModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes,
        classification="multilabel",
        print_freq=100,
        batch_size=10,
        d_size=500,
        pretrained=False,
        channel_size=3,
    ):
        super().__init__()
        self._d_size = d_size
        self._b_size = batch_size
        self._print_freq = print_freq
        self.model_name = model_name
        self.num_classes = num_classes
        self.classification = classification
        self.model = eval(f"models.{model_name}()")
        if pretrained:
            print(f"Using pretrained {model_name} model")
            state_dict = torch.load("densenet121-a639ec97.pth")
            state_dict = self.update_densenet_keys(state_dict)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Using {model_name} model from scratch")
        self.model.features[0] = nn.Conv2d(channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # freeze_model(self.model)
        if "resnet" in model_name.lower():
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        elif "densenet" in model_name.lower():
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, self.num_classes)
        elif "vit_b_16" in model_name.lower():
            num_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(num_features, self.num_classes)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
        
    def update_densenet_keys(self, state_dict):
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return state_dict

    def remove_head(self):
        if "resnet" in self.model_name.lower():
            num_features = self.model.fc.in_features
            id_layer = nn.Identity(num_features)
            self.model.fc = id_layer
        elif "densenet" in self.model_name.lower():
            num_features = self.model.classifier.in_features
            id_layer = nn.Identity(num_features)
            self.model.classifier = id_layer
        elif "vit" in self.model_name.lower():
            num_features = self.model.heads.head.in_features
            id_layer = nn.Identity(num_features)
            self.model.heads.head = id_layer
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

    def forward(self, x, classify=True):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch["image"], batch["label"]

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        out = out.squeeze()
        if self.classification == "multilabel":
            prob = torch.sigmoid(out)
            if img.shape[0] == 1:
                prob = prob.unsqueeze(0)
            try:
                loss = F.binary_cross_entropy(prob, lab)
            except:
                raise
        elif self.classification == "multiclass":
            prob = F.softmax(out, dim=1)
            loss = F.cross_entropy(out, lab)
        else:
            raise NotImplementedError(
                f"Classification {self.classification} not implemented"
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        if (batch_idx % self._print_freq == 0) or (
            batch_idx == (int(self._d_size / self._b_size) - 2)
        ):
            print(f"batch {batch_idx} train_loss: {loss}")
        self.log("train_loss", loss)
        grid = torchvision.utils.make_grid(
            batch["image"][0:4, ...], nrow=2, normalize=True
        )
        #self.logger.experiment.add_image("images", grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        if (batch_idx % self._print_freq == 0) or (
            batch_idx == (int(self._d_size / self._b_size) - 2)
        ):
            print(f"batch {batch_idx} val_loss: {loss}")
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log("test_loss", loss)


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
        target = iteration["label"]
    else:
        raise ValueError("Something is wrong. __getitem__ must return list or dict")
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Get predictions
    with torch.no_grad():
        output = model(images, classify=True)

    return output, target


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
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


def process_feature(feature_type, start, hours, use_last_hours, recording_ids, channels_to_use, data_folder, patient_id):
    feature_data = {f'{feature_type}_{item}': [] for item in ['features', 'feature_names', 'sampling_frequency', 'utility_frequency', 'channels', 'recording_id', 'quality', 'hour']}

    for h in range(start, hours + start):
        if use_last_hours:
            h_to_use = -h
        features, feature_names, sampling_frequency, utility_frequency, channels, recording_id, quality, hour = get_recording_features(
            recording_ids=recording_ids, recording_id_to_use=h_to_use, data_folder=data_folder, patient_id=patient_id, group=feature_type.upper(), channels_to_use=channels_to_use)

        for item in feature_data.keys():
            feature_data[item].append(locals()[item.split('_', 1)[-1]])

    return feature_data


import numpy as np
from typing import List


def check_artifacts(epoch: np.ndarray, low_threshold: float=-250, high_threshold: float=250) -> bool:
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
    #if abs(np.std(epoch)) > 10 * abs(np.mean(epoch)):
    #    return True
    
    # Fast rising or decreasing signal amplitude (If the absolute difference between any two consecutive samples is above a threshold)
    #if np.any(np.abs(np.diff(epoch)) > 10 * np.mean(np.abs(np.diff(epoch)))):
    #    return True

    return False


def compute_score(signal: np.ndarray, signal_frequency: float, epoch_size: int, window_size: int, 
                  stride_length: int, high_threshold: float = None, low_threshold: float = None) -> Dict[int, float]:
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
    Dict[int, float]
        A dictionary where keys are the starting time (in seconds) of each window and values are the corresponding scores.
    """
    num_channels, num_observations = signal.shape
    epoch_samples = int(signal_frequency * epoch_size)
    window_samples = int(signal_frequency * window_size * 60)
    stride_samples = int(signal_frequency * stride_length * 60)
    scores = {}
    for start in range(0, num_observations - window_samples + 1, stride_samples):
        window = signal[:, start : start + window_samples]
        artifact_epochs = 0
        total_epochs = 0
        for epoch_start in range(0, window_samples - epoch_samples + 1, epoch_samples):
            for channel in range(num_channels):
                epoch = window[channel, epoch_start : epoch_start + epoch_samples]
                if check_artifacts(epoch):
                    artifact_epochs += 1
                    break  # if any channel in the epoch has artifact, consider the whole epoch contaminated
            total_epochs += 1
        score = 1 - artifact_epochs / total_epochs
        # Converting start sample index to time in seconds
        start_time = start / signal_frequency
        scores[start_time] = score
    return scores


def keep_best_window(signal: np.ndarray, scores: Dict[int, float], signal_frequency: float, window_size: int) -> np.ndarray:
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
    
    Returns
    -------
    np.ndarray
        The EEG signal where only the window with the best score is kept and all other samples are set to NaN.
    """
    num_channels, num_observations = signal.shape
    window_samples = int(signal_frequency * window_size * 60)
    
    # Find the window with the best score
    best_start_time = max(scores, key=scores.get)
    best_start_sample = int(best_start_time * signal_frequency)
    
    # Create a new signal array filled with NaN
    #new_signal = np.full((num_channels, num_observations), np.nan)
    
    # Keep only the window with the best score
    #new_signal[:, best_start_sample : best_start_sample + window_samples] = signal[:, best_start_sample : best_start_sample + window_samples]

    new_signal = signal[:, best_start_sample : best_start_sample + window_samples]
    
    return new_signal


def load_recording_data_wrapper(record_name):
    """
    Loads the EEG signal of a recording and applies artifact removal.
    """

    window_size = 5 # minutes
    stride_length = 1 # minutes
    epoch_size = 10 # seconds

    signal_data, signal_channels, sampling_frequency = load_recording_data(record_name)

    if FILTER_SIGNALS:
        scores = compute_score(signal_data, signal_frequency=sampling_frequency, epoch_size=epoch_size, window_size=window_size, stride_length=stride_length)
        signal_data_filtered = keep_best_window(signal=signal_data, scores=scores, signal_frequency=sampling_frequency, window_size=window_size)
        signal_return = signal_data_filtered

        # Save the signal, scores and filtered signal of the first channel as pandas dataframes
        df_signal = pd.DataFrame(signal_data[0])
        df_signal.columns = ['signal']
        df_signal['time'] = df_signal.index / sampling_frequency
        df_signal['score'] = df_signal['time'].apply(lambda x: scores.get(x, np.nan))
        #df_signal.to_csv(f'./data/01_intermediate_analysis/{record_name.split("/")[-1]}.csv', index=False)
    else:
        signal_return = signal_data

    return signal_return, signal_channels, sampling_frequency
