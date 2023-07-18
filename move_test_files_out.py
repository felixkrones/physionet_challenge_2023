""" Move all files from file into new folder"""
import os
import sys
import pandas as pd
from tqdm import tqdm


def move_files(source_dir,target_dir,split_file,test_split):

    split_pd = pd.read_csv(split_file, dtype={"split": int, "patient_id": str})
    test_pd = split_pd[split_pd["split"]==test_split]
    test_patients = test_pd["patient_id"].tolist()

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    print(f"----- Moving {len(test_patients)} test files of split {test_split} out. -----")
    for patient in tqdm(test_patients):
        os.rename(os.path.join(source_dir, patient), os.path.join(target_dir, patient))


if __name__ == '__main__':
    print("------------- move_test_files_out.py -------------")

    source_dir = "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training"
    target_dir = "/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing"
    split_file = "/data/inet-multimodal-ai/wolf6245/src/physionet_challenge_2023/data/splits.csv"
    split_column = str(sys.argv[2]) # "split" "hospital"
    test_split = int(sys.argv[1])

    move_files(source_dir,target_dir,split_file,test_split)

