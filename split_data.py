import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm


def run_split(input_dir, output_dir, random_seed, test_ratio, cv:bool=False):
    # Get all folders in input_dir as list
    folders = [f for f in input_dir.glob('*/') if f.is_dir()]
    print(f"Patients: {len(folders)}")

    # Split folders randomly into train, validation and test
    np.random.seed(random_seed)
    folders.sort()
    np.random.shuffle(folders)
    test_size = int(len(folders) * test_ratio)
    print(f"test_size: {test_size}")

    if cv:
        print(f"Running cv split {(1/test_ratio)} times ---------------")
        if not (1/test_ratio).is_integer():
            raise ValueError("test_ratio must be an integer fraction of 1")
        test_folders_list = []
        train_folders_list = []
        name_list = []
        for i in range(int(1/test_ratio)):
            test_start = (test_size*i)
            test_end = (test_size*(i+1))
            train_start = (test_size*(i+1))
            train_end = (test_size*(i+2))
            if i == (1/test_ratio)-2:
                train_end = len(folders)
            if i == (1/test_ratio)-1:
                train_start = 0
                train_end = test_size
                test_end = len(folders)
            test_folders_list.append(folders[test_start:test_end])
            train_folders_list.append(folders[train_start:train_end])
            name_list.append(i)
            print(f"Name: {i}, test: ({test_start},{test_end}), len: {len(folders[test_start:test_end])}, train: ({train_start},{train_end}), len: {len(folders[train_start:train_end])}")
    else:
        print("Running one seed split ---------------")
        test_folders_list = [folders[:test_size]]
        train_folders_list = [folders[test_size:]]
        name_list = [random_seed]
    for name, test_folders, train_folders in zip(name_list, test_folders_list, train_folders_list):
        print(f"Start copying test folders {name} out of {len(name_list)}...")
        (output_dir / Path(f"test_{name}")).mkdir(parents=True, exist_ok=True)
        for folder in tqdm(test_folders):
            if not os.path.exists(os.path.join(f"{output_dir}/test_{name}", str(folder).split("/")[-1])):
                os.system(f"cp -r {folder} {output_dir}/test_{name}")
        print(f"Finished copying test folders, starting with train folders {name} out of {len(name_list)}...")
        (output_dir / Path(f"train_{name}")).mkdir(parents=True, exist_ok=True)
        for folder in tqdm(train_folders):
            if not os.path.exists(os.path.join(f"{output_dir}/train_{name}", str(folder).split("/")[-1])):
                os.system(f"cp -r {folder} {output_dir}/train_{name}")


if __name__ == '__main__':
    # Set locations and parameters
    input_dir = Path("/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023/physionet.org/files/i-care/1.0/training")
    output_dir = Path("/data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2023")

    random_seed = int(sys.argv[1])
    test_ratio = float(sys.argv[2])
    cv = bool(sys.argv[3])
    print(f"Start splitting for random_seed {random_seed} with cv {cv}")

    run_split(input_dir, output_dir, random_seed, test_ratio, cv)
