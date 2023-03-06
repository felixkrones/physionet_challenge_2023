#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

import os
from helper_code import *
import librosa
import pandas as pd
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import timm
import torch
from torch import nn
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    batch_size = 64
    val_size = 0.2
    max_hours = 72
    min_quality = 0.0
    max_epochs = 2

    # Get device
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    accelerator = "cpu" # "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")

    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    #TODO: Implement split into train and validation set
    num_val = int(num_patients * val_size)
    num_train = num_patients - num_val
    train_ids = patient_ids[:num_train]
    val_ids = patient_ids[num_train:]

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    # Get DL data
    train_dataset = EEGDataset(data_folder, patient_ids = train_ids, device=device)
    val_dataset = EEGDataset(data_folder, patient_ids = val_ids, device=device)
    torch_dataset = EEGDataset(data_folder, patient_ids = patient_ids, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    data_loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # Find last checkpoint
    resume_from_checkpoint = get_last_chkpt(model_folder)

    # Train torch model
    #checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last" #TODO: Check if this is necessary
    torch_model = get_tv_model(batch_size=batch_size, d_size=len(train_loader)).to(device)
    trainer = pl.Trainer(
        accelerator=accelerator,
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_last=True, save_top_k=1)],
        log_every_n_steps=1,
        max_epochs=max_epochs,
        gpus=1,
        enable_progress_bar=True,
        logger=TensorBoardLogger(model_folder, name=''),
        resume_from_checkpoint=resume_from_checkpoint,
    )
    trainer.logger._default_hp_metric = False
    print("Start training torch model...")
    trainer.fit(torch_model, train_loader, val_loader)
    print("Finished training torch model. Now calculating predictions...")
    
    output_list, patient_id_list, hour_list, quality_list = torch_prediction(torch_model, data_loader, device)
    

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)
        agg_outcome_probability_torch, outcome_probabilities_torch = torch_predictions_for_patient(output_list, patient_id_list, hour_list, quality_list, patient_id, max_hours=max_hours, min_quality=min_quality)

        # Extract features.
        current_features = get_features(patient_metadata, recording_metadata, recording_data, max_hours=max_hours, min_quality=min_quality)
        current_features = np.hstack((current_features, outcome_probabilities_torch)) #TODO: Optional: Add torch model predictions
        #current_features = outcome_probabilities_torch
        features.append(current_features)

        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators = 123  # Number of trees in the forest.
    max_leaf_nodes = None  # Maximum number of leaf nodes in each tree.
    max_depth = 8
    random_state = 42  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    # Train the models.
    print("Start training rf models...")
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model, torch_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    model = joblib.load(filename)
    file_path = os.path.join(model_folder, 'checkpoint.pth')
    #file_path = os.path.join(model_folder, 'last.ckpt')
    model["torch_model"] = load_last_pt_ckpt(file_path)
    return model

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    torch_model = models['torch_model']

    # Get device
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Torch prediction
    data_set = EEGDataset(data_folder, patient_ids = [patient_id], device = device)
    data_loader = DataLoader(data_set, batch_size=1, num_workers=4, shuffle=False)
    output_list, patient_id_list, hour_list, quality_list = torch_prediction(torch_model, data_loader, device)
    agg_outcome_probability_torch, outcome_probabilities_torch = torch_predictions_for_patient(output_list, patient_id_list, hour_list, quality_list, patient_id) #TODO: Think about whether to use max_hours and min_quality.

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data) #TODO: Think about whether to use max_hours and min_quality.
    features = np.hstack((features, outcome_probabilities_torch)) #TODO: Optional: Use the torch model to predict the outcome probability.
    #features = np.array([outcome_probabilities_torch])
    features = features.reshape(1, -1)

    # Impute missing data.
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
                last_version = max(last_version, int(folder.split("_")[-1]))
        last_version = f"version_{last_version}"
        if os.path.isfile(f"{model_folder}/{last_version}/checkpoints/last.ckpt"):
            resume_from_checkpoint = f"{model_folder}/{last_version}/checkpoints/last.ckpt"
        else:
            resume_from_checkpoint = None
    else:
        resume_from_checkpoint = None

    if resume_from_checkpoint is not None:
        print("Resuming from checkpoint: ", resume_from_checkpoint)
    else:
        print("No checkpoint found. Starting from scratch.")
    
    return resume_from_checkpoint


def torch_prediction(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        output_list = []
        patient_id_list = []
        hour_list = []
        quality_list = []
        for _, batch in enumerate(tqdm(data_loader)):
            data, targets, ids, hours, qualities = batch["image"], batch["label"], batch["id"], batch["hour"], batch["quality"]
            data = data.to(device)
            outputs = model(data)
            outputs = torch.sigmoid(outputs) #TODO: Check whether to convert using softmax
            output_list = output_list + outputs.cpu().numpy().tolist()
            patient_id_list = patient_id_list + ids
            hour_list = hour_list + hours
            quality_list = quality_list + qualities
    return output_list, patient_id_list, hour_list, quality_list


def torch_predictions_for_patient(output_list, patient_id_list, hour_list, quality_list, patient_id, max_hours=72, min_quality=0):
    # Get the predictions for the patient
    patient_mask = np.array([True if p == patient_id else False for p in patient_id_list])
    outcome_probabilities_torch = np.array(output_list)[patient_mask]
    hours_patients = np.array(hour_list)[patient_mask].astype(int).tolist()

    if len(outcome_probabilities_torch[0]) == 1:
        outcome_probabilities_torch = [i[0] for i in outcome_probabilities_torch]
    else:
        raise ValueError("The torch model should only predict one value per patient.")

    #  Get values for the first 72 hours
    # TODO: Think about if and how to fill missing hours or if to leave out
    outcome_probabilities_torch = [outcome_probabilities_torch[hours_patients.index(hour)] if hour in hours_patients else np.nan for hour in range(max_hours)]

    # Aggregate the probabilities
    agg_outcome_probability_torch = np.nanmax(outcome_probabilities_torch) #TODO: Choose a good aggregation method, maybe max?

    return agg_outcome_probability_torch, outcome_probabilities_torch


def get_tv_model(model_name="densenet121", num_classes=1, batch_size=64, d_size=500):
    model = torchvisionModel(
            model_name=model_name,
            num_classes=num_classes,
            print_freq=250,
            batch_size=batch_size,
            d_size=d_size,
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
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model, torch_model=None):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)
    if torch_model is not None:
        file_path = os.path.join(model_folder, 'checkpoint.pth')
        torch.save({"model": torch_model.state_dict()}, file_path)

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data, return_as_dict=False, max_hours=73, min_quality=0.0):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
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

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])
    patient_features_dict = {"age": age, "female": female, "male": male, "other": other, "rosc": rosc, "ohca": ohca, "vfib": vfib, "ttm": ttm}

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel over all recordings
    available_signal_data = list()
    hours = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            quality = get_quality_scores(recording_metadata)[i]
            hour = get_hours(recording_metadata)[i]
            if (quality >= min_quality) and (hour <= max_hours):
                signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
                available_signal_data.append(signal_data)
                hours.append(hour)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel 
    # of the MOST RECENT recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            quality = get_quality_scores(recording_metadata)[i]
            hour = get_hours(recording_metadata)[i]
            if (quality >= min_quality) and (hour <= max_hours):
                index = i
                break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))
    recording_features_dict = {"quality_score": quality_score}
    for s in ["signal_mean", "signal_std", "delta_psd_mean", "theta_psd_mean", "alpha_psd_mean", "beta_psd_mean"]:
        for i, c in enumerate(channels):
            recording_features_dict[s + "_" + c] = eval(s)[i]

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))
    features_dict = patient_features_dict
    features_dict.update(recording_features_dict)
    features_dict.update({"max_hours": np.max(hours)})
    features = np.hstack((features, np.max(hours)))

    if return_as_dict:
        return features_dict
    else:
        return features


class EEGDataset(Dataset):
    def __init__(
        self, data_folder, patient_ids, device
    ):
        self.channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
            'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']

        recording_locations_list = list()
        patient_ids_list = list()
        labels_list = list()
        hours_list = list()
        qualities_list = list()
        for patient_id in patient_ids:
            patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
            recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')
            patient_metadata = load_text_file(patient_metadata_file)
            recording_metadata = load_text_file(recording_metadata_file)
            recording_ids = get_column(recording_metadata, 'Record', str)
            hours = get_column(recording_metadata, 'Hour', str)
            qualities = get_column(recording_metadata, 'Quality', str)
            current_outcome = get_outcome(patient_metadata)
            for recording_id, hour, quality in zip(recording_ids, hours, qualities):
                if recording_id != 'nan':
                    recording_location = os.path.join(data_folder, patient_id, recording_id)
                    recording_locations_list.append(recording_location)
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
        signal_data, sampling_frequency, signal_channels = load_recording(self.recording_locations[idx])
        signal_data = reorder_recording_channels(signal_data, signal_channels, self.channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
        signal_data = librosa.feature.melspectrogram(y=signal_data, sr=100, n_mels=224)
        signal_data = torch.from_numpy(signal_data)
        signal_data = nn.functional.normalize(signal_data).to(self._precision) #TODO: Check if this is the right way to normalize the signal data.
        id = self.patient_ids[idx]
        label = self.labels[idx]
        hour = self.hours[idx]
        quality = self.qualities[idx]

        signal_data = signal_data.to(self.device)
        label = torch.from_numpy(np.array(label)).to(self._precision).to(self.device)

        return {"image": signal_data, "label": label, "id": id, "hour": hour, "quality": quality}


class torchvisionModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes,
        classification="multilabel",
        print_freq=100,
        batch_size=10,
        d_size=500,
    ):
        super().__init__()
        self._d_size = d_size
        self._b_size = batch_size
        self._print_freq = print_freq
        self.model_name = model_name
        self.num_classes = num_classes
        self.classification = classification
        self.model = eval(f"models.{model_name}(weights='DEFAULT')")
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
