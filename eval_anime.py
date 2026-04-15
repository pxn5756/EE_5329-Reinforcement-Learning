import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import tkinter as tk
from tkinter import filedialog

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import shutil


def main():

    # Open file dialog to select model
    root = tk.Tk()
    root.withdraw()  # hides the empty tkinter window

    model_path = filedialog.askopenfilename(
        title="Select Model",
        initialdir="model/",
        filetypes=[("Zip files", "*.zip")]
    )

    if not model_path:
        print("No model selected. Exiting.")
        exit()  
    else:
        model = TD3.load(model_path)

    head, tail = os.path.split(model_path)
  
    # Create log dir
    log_dir = head
    os.makedirs(log_dir + "/eval", exist_ok=True)

    # Initialise the environment
    eval_env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    # Reset the environment to generate the first observation
    observation, info = eval_env.reset(seed=42)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    eval_env.close()
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


if __name__ == "__main__":
    main()