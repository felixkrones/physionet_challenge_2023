""" Move all files from file into new folder"""
import os
import pandas as pd
from tqdm import tqdm


def move_files(source_dir,target_dir):
   all_dirs = os.listdir(target_dir)
   print(f"----- Moving {len(all_dirs)} test files back in. -----")
   for patient in tqdm(all_dirs):
      os.rename(os.path.join(target_dir, patient), os.path.join(source_dir, patient))


if __name__ == '__main__':
   print("------------- move_test_files_back.py -------------")

   source_dir = "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/training"
   target_dir = "/Users/felixkrones/python_projects/data/physionet_challenge_2023/physionet.org/files/i-care/2.0/testing"

   move_files(source_dir,target_dir)
