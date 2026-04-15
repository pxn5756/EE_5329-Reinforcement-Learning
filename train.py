
import os
from pyexpat import model

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import shutil
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, exp_name: str,verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, exp_name+ "_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True
    

def train(model, name="baseline", log_dir="model/baseline"):
    
    start_wall = time.perf_counter()

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, exp_name=name)

    # Train the agent
    model.learn(total_timesteps=int(100000), callback=callback)

    end_wall = time.perf_counter()
    print(f"Training time: {end_wall - start_wall:.2f} seconds")
    model.save(log_dir + "/" + name + "_model")

    shutil.rmtree(log_dir + "/" + name + "_model")


    results = load_results(log_dir)
    x, y = ts2xy(results, "timesteps")

    # Plot
    fig = plt.figure('Rewards')
    plt.plot(x, y, label="Learning Curve")

    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title("Learning Curve")
    fig_name = log_dir + "/learning_curve.png"
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")

def main():
    seed = 42
    experiment_name = "experiment1"

    # Create and wrap the environment
    env = gym.make("LunarLanderContinuous-v3")

    log_dir = "model/" + experiment_name
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir + "/train_monitor.csv")

    # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Run baseline experiment
    # model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, seed=seed, device="auto")
    # train(model, name=experiment_name, log_dir=log_dir)

    # Experiment 1: Increase gradient steps
    model= TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, seed=seed, device="auto",
            gradient_steps=3)
    train(model, name=experiment_name, log_dir=log_dir)


if __name__ == "__main__":
    main()