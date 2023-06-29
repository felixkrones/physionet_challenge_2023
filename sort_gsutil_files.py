import os
from tqdm import tqdm

source_dir = "/Users/felixkrones/python_projects/data/physionet_challenge_2023/gsutil"
target_dir = '/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training'

# For all files in the source_dir, move them into the subfolders of the target_dir based on the first 4 characters of the filename
for filename in tqdm(os.listdir(source_dir)):
    patient_id = filename[:4]
    target_dir_patient = os.path.join(target_dir, patient_id)
    os.rename(os.path.join(source_dir, filename), os.path.join(target_dir_patient, filename))
