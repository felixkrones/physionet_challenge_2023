#! /bin/bash

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/envs/physionet
conda create --prefix $CONPREFIX

# Activate your environment
source activate $CONPREFIX

# Install packages...
conda install -c anaconda pillow
conda install -c anaconda pandas
conda install -c anaconda numpy
conda install -c anaconda h5py
conda install -c anaconda scikit-image
conda install -c conda-forge matplotlib
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install -c conda-forge timm
conda install -c conda-forge imageio