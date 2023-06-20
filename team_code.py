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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


ECG_CHANNELS = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
EEG_CHANNELS = ['F3', 'P3', 'F4', 'P4']
BIPOLAR_MONTAGES = [('F3','P3'), ('F4','P4')] #TODO: Implement for torch as well
USE_TORCH = False
PARAMS_DEVICE = {"num_workers": 20} #os.cpu_count()}
print(f"CPU count: {os.cpu_count()}")
print(PARAMS_DEVICE)

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Parameters
    params_torch = {'batch_size': 16, 'val_size': 0.3, 'max_epochs': 10, 'pretrained': True, 'devices': 1, 'num_nodes': 1}
    c_model = "rf" # "xgb" or "rf
    params_rf = {'n_estimators': 123, 'max_depth': 8, 'max_leaf_nodes': None, 'random_state': 42, 'n_jobs': 8}
    params_xgb = {'max_depth': 8, 'eval_metric': 'auc', 'nthread': 8}

    if USE_TORCH:
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

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device} and accelerator {accelerator}")

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
    if USE_TORCH:
        # Split into train and validation set
        num_val = int(num_patients * params_torch['val_size'])
        num_train = num_patients - num_val
        patient_ids_aux = patient_ids.copy()
        random.shuffle(patient_ids_aux) #TODO: Add back in
        train_ids = patient_ids_aux[:num_train]
        val_ids = patient_ids_aux[num_train:]

        # Get EEG DL data
        train_dataset_eeg = EEGDataset(data_folder, patient_ids = train_ids, device=device, group = "eeg")
        val_dataset_eeg = EEGDataset(data_folder, patient_ids = val_ids, device=device, group = "eeg")
        torch_dataset_eeg = EEGDataset(data_folder, patient_ids = patient_ids, device=device, group = "eeg")
        train_loader_eeg = DataLoader(train_dataset_eeg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=True, pin_memory=True)
        val_loader_eeg = DataLoader(val_dataset_eeg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)
        data_loader_eeg = DataLoader(torch_dataset_eeg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)

        # Get ECG DL data
        train_dataset_ecg = EEGDataset(data_folder, patient_ids = train_ids, device=device, group = "ecg")
        val_dataset_ecg = EEGDataset(data_folder, patient_ids = val_ids, device=device, group = "ecg")
        torch_dataset_ecg = EEGDataset(data_folder, patient_ids = patient_ids, device=device, group = "ecg")
        train_loader_ecg = DataLoader(train_dataset_ecg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=True, pin_memory=True)
        val_loader_ecg = DataLoader(val_dataset_ecg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)
        data_loader_ecg = DataLoader(torch_dataset_ecg, batch_size=params_torch['batch_size'], num_workers=PARAMS_DEVICE["num_workers"], shuffle=False, pin_memory=True)

        # Find last checkpoint
        model_folder_eeg = os.path.join(model_folder, "eeg")
        model_folder_ecg = os.path.join(model_folder, "ecg")
        checkpoint_path_eeg = get_last_chkpt(model_folder_eeg)
        checkpoint_path_ecg = get_last_chkpt(model_folder_ecg)

        # Define torch model
        torch_model_eeg = get_tv_model(batch_size=params_torch['batch_size'], d_size=len(train_loader_eeg), pretrained=params_torch['pretrained'])
        torch_model_ecg = get_tv_model(batch_size=params_torch['batch_size'], d_size=len(train_loader_ecg), pretrained=params_torch['pretrained'])
        trainer_eeg = pl.Trainer(
            accelerator=accelerator,
            devices=params_torch["devices"],
            num_nodes=params_torch["num_nodes"],
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_last=True, save_top_k=1)],
            log_every_n_steps=1,
            max_epochs=params_torch['max_epochs'],
            enable_progress_bar=True,
            logger=TensorBoardLogger(model_folder_eeg, name=''),
        )
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
        trainer_eeg.logger._default_hp_metric = False
        trainer_ecg.logger._default_hp_metric = False

        # Train EEG torch model
        print("Start training EEG torch model...")
        start_time_torch = time.time()
        trainer_eeg.fit(torch_model_eeg, train_loader_eeg, val_loader_eeg, ckpt_path=checkpoint_path_eeg)
        print(f"Finished training EEG torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min.")

        # Train ECG torch model
        print("Start training ECG torch model...")
        start_time_torch = time.time()
        trainer_ecg.fit(torch_model_ecg, train_loader_ecg, val_loader_ecg, ckpt_path=checkpoint_path_ecg)
        print(f"Finished training ECG torch model for {params_torch['max_epochs']} epochs after {round((time.time()-start_time_torch)/60,4)} min.")

        # Get EEG predictions
        print("Start predicting EEG torch features ...")
        output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg = torch_prediction(torch_model_eeg, data_loader_eeg, device)
        print("Done with EEG torch.")

        # Get ECG predictions
        print("Start predicting ECG torch features ...")
        output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg = torch_prediction(torch_model_ecg, data_loader_ecg, device)
        print("Done with ECG torch.")

        print("Done with torch")
        
    print("Calculating features...")
    features = list()
    feature_names = list()
    outcomes = list()
    patients = list()
    cpcs = list()
    patient_ids_aux = list()
    outcome_probabilities_torch_eeg_aux = list()
    outcome_probabilities_torch_ecg_aux = list()
    outcome_flags_torch_eeg_aux = list()
    outcome_flags_torch_ecg_aux = list()
    for i in tqdm(range(num_patients)):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data and extract features
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_features, current_feature_names = get_features(data_folder, patient_ids[i])

        if USE_TORCH:
            # Get torch predictions
            agg_outcome_probabilities_torch_eeg, outcome_probabilities_torch_eeg, outcome_flags_torch_eeg = torch_predictions_for_patient(output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg, patient_ids[i])
            agg_outcome_probabilities_torch_ecg, outcome_probabilities_torch_ecg, outcome_flags_torch_ecg = torch_predictions_for_patient(output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg, patient_ids[i])
            current_features = np.hstack((current_features, outcome_probabilities_torch_eeg)) #TODO: Combine new torch features, e.g. add outcome_flags_torch
            current_features = np.hstack((current_features, outcome_probabilities_torch_ecg))
            current_feature_names = np.hstack((current_feature_names, [f"prob_eeg_torch_{i}" for i in range(len(outcome_probabilities_torch_eeg))]))
            current_feature_names = np.hstack((current_feature_names, [f"prob_ecg_torch_{i}" for i in range(len(outcome_probabilities_torch_ecg))]))
            outcome_probabilities_torch_eeg_aux.append(outcome_probabilities_torch_eeg)
            outcome_probabilities_torch_ecg_aux.append(outcome_probabilities_torch_ecg)
            outcome_flags_torch_eeg_aux.append(outcome_flags_torch_eeg)
            outcome_flags_torch_ecg_aux.append(outcome_flags_torch_ecg)

        # Extract labels.
        features.append(current_features)
        feature_names.append(current_feature_names)
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)
        patients.append(patient_ids[i])

    features = np.vstack(features)
    feature_names = np.vstack(feature_names)
    outcomes = np.vstack(outcomes)
    patients = np.vstack(patients)
    cpcs = np.vstack(cpcs)

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

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
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model, torch_model_eeg=torch_model_eeg, torch_model_ecg=torch_model_ecg)

    # Plot and save feature importance
    feature_importance = outcome_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(12, len(feature_importance)/2))
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
    if USE_TORCH:
        model["torch_model_eeg"] = load_last_pt_ckpt(file_path_eeg)
        model["torch_model_ecg"] = load_last_pt_ckpt(file_path_ecg)
    else:
        model["torch_model_eeg"] = None
        model["torch_model_ecg"] = None
    return model

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):

    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    torch_model_eeg = models['torch_model_eeg']
    torch_model_ecg = models['torch_model_ecg']

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load data.
    features, _ = get_features(data_folder, patient_id) 

    # Torch prediction
    if USE_TORCH:
        data_set_eeg = EEGDataset(data_folder, patient_ids = [patient_id], device = device, load_labels = False, group="eeg")
        data_set_ecg = EEGDataset(data_folder, patient_ids = [patient_id], device = device, load_labels = False, group="ecg")
        data_loader_eeg = DataLoader(data_set_eeg, batch_size=1, num_workers=PARAMS_DEVICE["num_workers"], shuffle=False)
        data_loader_ecg = DataLoader(data_set_ecg, batch_size=1, num_workers=PARAMS_DEVICE["num_workers"], shuffle=False)
        output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg = torch_prediction(torch_model_eeg, data_loader_eeg, device)
        output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg = torch_prediction(torch_model_ecg, data_loader_ecg, device)
        agg_outcome_probability_torch_eeg, outcome_probabilities_torch_eeg, outcome_flags_torch_eeg = torch_predictions_for_patient(output_list_eeg, patient_id_list_eeg, hour_list_eeg, quality_list_eeg, patient_id)
        agg_outcome_probability_torch_ecg, outcome_probabilities_torch_ecg, outcome_flags_torch_ecg = torch_predictions_for_patient(output_list_ecg, patient_id_list_ecg, hour_list_ecg, quality_list_ecg, patient_id)
        features = np.hstack((features, outcome_probabilities_torch_eeg)) #TODO: Combine new torch features, e.g. add outcome_flags_torch
        features = np.hstack((features, outcome_probabilities_torch_ecg))

    # Impute missing data.
    features = features.reshape(1, -1)
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
            hour_list = hour_list + hours
            quality_list = quality_list + qualities
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
    outcome_probabilities_torch_imputed = pd.Series(outcome_probabilities_torch, dtype=object).fillna(0).tolist()
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


def get_tv_model(model_name="densenet121", num_classes=1, batch_size=64, d_size=500, pretrained=False):
    model = torchvisionModel(
            model_name=model_name,
            num_classes=num_classes,
            print_freq=250,
            batch_size=batch_size,
            d_size=d_size,
            pretrained=pretrained
        )
    return model


# Load last checkpoint
def load_last_pt_ckpt(ckpt_path):
    model = get_tv_model()
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
            file_path = os.path.join(torch_model_folder, 'checkpoint.pth')
            torch.save({"model": torch_model.state_dict()}, file_path)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

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

# Load recording data. #TODO: All use all recordings
def get_recording_features(recording_ids, recording_id_to_use, data_folder, patient_id, group, channels_to_use):
    if group == "EEG":
        dummy_channels = 8
    elif group == "ECG":
        dummy_channels = 10
    else:
        raise ValueError("Group should be either ECG or EEG")
    if len(recording_ids) > 0:
        recording_id = recording_ids[recording_id_to_use]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')
            if all(channel in channels for channel in channels_to_use) or group == "ECG":
                data, channels = reduce_channels(data, channels, channels_to_use)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                if group == "EEG":
                    if BIPOLAR_MONTAGES is not None:
                        data = np.array([data[channels.index(montage[0]), :] - data[channels.index(montage[1]), :] for montage in BIPOLAR_MONTAGES])
                    recording_features, recording_feature_names = get_eeg_features(data, sampling_frequency)
                elif group == "ECG":
                    features, recording_feature_names_aux = get_ecg_features(data)
                    recording_features = expand_channels(features, channels, channels_to_use).flatten()
                    if channels == channels_to_use:
                        recording_feature_names = recording_feature_names_aux.flatten()
                    else:
                        num_current_channels, num_samples = np.shape(recording_feature_names_aux)
                        num_requested_channels = len(channels_to_use)
                        recording_feature_names = np.empty((num_requested_channels, num_samples), "S15")
                        for i, channel in enumerate(channels_to_use):
                            if channel in channels:
                                j = channels.index(channel)
                                recording_feature_names[i, :] = [f"{n}_{i}" for n in recording_feature_names_aux[j, :]]
                            else:
                                recording_feature_names[i, :] = f'ECG_nan_{i}'
                        recording_feature_names = recording_feature_names.flatten()
                else:
                    raise NotImplementedError(f"Group {group} not implemented.")
            else:
                print(f"For patient {patient_id} recording {recording_id} the channels {channels_to_use} are not all available.")
                recording_features = float('nan') * np.ones(dummy_channels) # 2 bipolar channels * 4 features / channel
                recording_feature_names = [f'{group}_nan_{i}' for i in range(dummy_channels)]
        else:
            recording_features = float('nan') * np.ones(dummy_channels) # 2 bipolar channels * 4 features / channel
            recording_feature_names = [f'{group}_nan_{i}' for i in range(dummy_channels)]
    else:
        recording_features = float('nan') * np.ones(dummy_channels) # 2 bipolar channels * 4 features / channel
        recording_feature_names = [f'{group}_nan_{i}' for i in range(dummy_channels)]

    return recording_features, recording_feature_names



# Extract features from the data.
def get_features(data_folder, patient_id, return_as_dict=False):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features, patient_feature_names = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_features, eeg_feature_names = get_recording_features(recording_ids=recording_ids, recording_id_to_use=-1, data_folder=data_folder, patient_id=patient_id, group="EEG", channels_to_use=EEG_CHANNELS)

    # Extract ECG features.
    ecg_features, ecg_feature_names = get_recording_features(recording_ids=recording_ids, recording_id_to_use=0, data_folder=data_folder, patient_id=patient_id, group="ECG", channels_to_use=ECG_CHANNELS)

    # Extract features.
    feature_values = np.hstack((patient_features, eeg_features, ecg_features))
    feature_names = np.hstack((patient_feature_names, eeg_feature_names, ecg_feature_names))

    if return_as_dict:
        return {k: v for k, v in zip(feature_names, feature_values)}
    else:
        return feature_values, feature_names

# Extract patient features from the data.
def get_patient_features(data):
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

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))
    feature_names = ["age", "female", "male", "other", "rosc", "ohca", "shockable_rhythm", "ttm"]

    return features, feature_names

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
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
    feature_names = np.array((np.array([f"delta_psd_mean_{i}" for i in range(data.shape[0])]), np.array([f"theta_psd_mean_{i}" for i in range(data.shape[0])]), np.array([f"alpha_psd_mean_{i}" for i in range(data.shape[0])]), np.array([f"beta_psd_mean_{i}" for i in range(data.shape[0])]))).T.flatten()

    return features, feature_names

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
    feature_names = np.array((np.array([f"ECG_mean_{i}" for i in range(data.shape[0])]), np.array([f"ECG_std_{i}" for i in range(data.shape[0])]))).T

    return features, feature_names


class EEGDataset(Dataset):
    def __init__(
        self, data_folder, patient_ids, device, group = "EEG", load_labels: bool=True
    ):
        
        self.group = group
        if self.group == "EEG":
            self.channels = EEG_CHANNELS
        elif self.group == "ECG":
            self.channels = ECG_CHANNELS
        else:
            raise NotImplementedError(f"Group {self.group} not implemented.")

        recording_locations_list = list()
        patient_ids_list = list()
        labels_list = list()
        hours_list = list()
        qualities_list = list()
        for patient_id in patient_ids:
            patient_metadata = load_challenge_data(data_folder, patient_id)
            recording_ids = find_recording_files(data_folder, patient_id)
            recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')
            recording_metadata = load_text_file(recording_metadata_file)
            hours = get_variable(recording_metadata, 'Hour', str)
            qualities = get_variable(recording_metadata, 'Quality', str)
            if load_labels:
                current_outcome = get_outcome(patient_metadata)
            else:
                current_outcome = 0
            for recording_id, hour, quality in zip(recording_ids, hours, qualities):
                if not is_nan(recording_id):
                    recording_locations_list.append(os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, self.group)))
                    patient_ids_list.append(patient_id)
                    labels_list.append(current_outcome)
                    hours_list.append(hour)
                    qualities_list.append(quality)
    
        self.recording_locations = recording_locations_list
        self.patient_ids = patient_ids_list
        self.labels = labels_list
        self.hours = hours_list
        self.qualities = qualities_list
        self.device = device
        self._precision = torch.float32

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal_data, signal_channels, sampling_frequency = load_recording_data(self.recording_locations[idx])
        utility_frequency = get_utility_frequency(self.recording_locations[idx] + '.hea')
        signal_data, signal_channels = reduce_channels(signal_data, signal_channels, self.channels)
        signal_data, sampling_frequency = preprocess_data(signal_data, sampling_frequency, utility_frequency)
        spectrograms = librosa.feature.melspectrogram(y=signal_data, sr=100, n_mels=224)
        spectrograms = torch.from_numpy(spectrograms)
        spectrograms = nn.functional.normalize(spectrograms).to(self._precision)
        id = self.patient_ids[idx]
        label = self.labels[idx]
        hour = self.hours[idx]
        quality = self.qualities[idx]

        label = torch.from_numpy(np.array(label)).to(self._precision)

        return {"image": spectrograms, "label": label, "id": id, "hour": hour, "quality": quality}


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
        self.model.features[0] = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
            loss = F.binary_cross_entropy(prob, lab)
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