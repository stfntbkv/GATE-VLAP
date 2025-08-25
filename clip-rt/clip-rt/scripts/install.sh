#!/bin/bash

set -e 
set -o pipefail

# Check if conda env 'clip-rt' exists
if conda info --envs | grep -q '^clip-rt'; then
    echo "Conda environment 'clip-rt' already exists, skipping creation."
else
    echo "Creating conda environment 'clip-rt'..."
    conda create -n clip-rt python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clip-rt
pip install uv


# Install open_clip for CLIP-RT
cd open_clip
uv pip install .
uv pip install 'open_clip_torch[training]'

cd ../libero

# Clone or update LIBERO repo
if [ -d "LIBERO/.git" ]; then
    cd LIBERO
    git pull
else
    echo "Cloning LIBERO repository..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO
fi

uv pip install . --system --upgrade
cd ..

# Install libero experiment requirements
uv pip install -r libero_requirements.txt --system --upgrade

uv pip install -U numpy==1.26.4