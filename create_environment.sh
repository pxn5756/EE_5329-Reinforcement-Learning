#!/bin/bash

python -m venv LunarLander
source LunarLander/Scripts/activate
python -m pip install --upgrade pip

pip install gymnasium[other]
pip install gymnasium[box2d]
pip install stable-baselines3[extra]

