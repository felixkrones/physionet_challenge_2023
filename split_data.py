import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm


def run_split(input_dir, output_dir, random_seed, test_ratio):
    # Get all folders in input_dir as list
    folders = [f for f in input_dir.glob('*/') if f.is_dir()]
    print(f"Patients: {len(folders)}")

    # Split folders randomly into train, validation and test
    np.random.seed(random_seed)
    folders.sort()
    np.random.shuffle(folders)
    test_size = int(len(folders) * test_ratio)
    test_folders = folders[:test_size]
    train_folders = folders[test_size:]
    print(f"test_size: {test_size}")

    # Copy folders to output_dir
    print("Start copying test folders...")
    (output_dir / Path(f"test_{random_seed}")).mkdir(parents=True, exist_ok=True)
    for folder in tqdm(test_folders):
        if not os.path.exists(os.path.join(f"{output_dir}/test_{random_seed}", str(folder).split("/")[-1])):
            os.system(f"cp -r {folder} {output_dir}/test_{random_seed}")
    print("Finished copying test folders, starting with train folders...")
    (output_dir / Path(f"train_{random_seed}")).mkdir(parents=True, exist_ok=True)
    for folder in tqdm(train_folders):
        if not os.path.exists(os.path.join(f"{output_dir}/train_{random_seed}", str(folder).split("/")[-1])):
            os.system(f"cp -r {folder} {output_dir}/train_{random_seed}")


if __name__ == '__main__':
    # Set locations and parameters
    input_dir = Path("/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/1.0/training")
    output_dir = Path("/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023")
    test_ratio = 0.1

    random_seed = int(sys.argv[1])
    print(f"Start splitting for random_seed {random_seed}")

    run_split(input_dir, output_dir, random_seed, test_ratio)
