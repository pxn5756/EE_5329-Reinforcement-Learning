#!/bin/bash

python -m venv LunarLander
source LunarLander/Scripts/activate
python -m pip install --upgrade pip

# For CUDA 13.0, use the following command to install PyTorch and torchvision:
# This allows for GPU to be used. NOTE: If you have a different CUDA version, please refer to the official PyTorch website for the correct installation command.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install gymnasium[other]
pip install gymnasium[box2d]
pip install stable-baselines3[extra]


