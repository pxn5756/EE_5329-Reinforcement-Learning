import gymnasium as gym
import numpy as np
import pygame
import os

def keyboard_demo_to_buffer(demo_dir="demos/"):
    """
    Control LunarLander using keyboard.
    Captures transitions (s, a, r, s', done) only.

    Controls:
        UP arrow    → main engine
        LEFT arrow  → left thruster
        RIGHT arrow → right thruster
        Q           → quit and save transitions
    """

    os.makedirs(demo_dir, exist_ok=True)

    # Initialize pygame for keyboard input only
    pygame.init()

    # Create environment
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    obs, _ = env.reset()

    # Storage for raw transitions only
    transitions = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "dones": []
    }

    total_reward = 0
    episode = 1
    running = True

    print("\nControls:")
    print("  UP    → main engine")
    print("  LEFT  → left thruster")
    print("  RIGHT → right thruster")
    print("  Q     → quit and save transitions")

    while running:
        action = np.array([0.0, 0.0])

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        # Get held keys
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            action[0] = 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[1] = -0.7
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[1] = 0.7
        if keys[pygame.K_w]:
            action[0] = 0.5
        if keys[pygame.K_e]:
            action[0] = 0.1

        # Store current obs before step
        current_obs = obs.copy()

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        transitions["observations"].append(current_obs)
        transitions["next_observations"].append(obs.copy())
        transitions["actions"].append(action.copy())
        transitions["rewards"].append(reward)
        transitions["dones"].append(done)

        total_reward += reward

        # Print to console
        print(f"\rEpisode: {episode} | Reward: {total_reward:.2f} | Transitions: {len(transitions['observations'])}", end="")

        # Reset on episode end
        if done:
            print(f"\nEpisode {episode} finished — reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0
            episode += 1

    env.close()
    pygame.quit()

    # Save raw transitions
    if len(transitions["observations"]) > 0:
        demo_path = os.path.join(demo_dir, "transitions.npz")
        np.savez(
            demo_path,
            observations=np.array(transitions["observations"]),
            next_observations=np.array(transitions["next_observations"]),
            actions=np.array(transitions["actions"]),
            rewards=np.array(transitions["rewards"]),
            dones=np.array(transitions["dones"])
        )
        print(f"\nSaved {len(transitions['observations'])} transitions to {demo_path}")

    return transitions


def load_transitions_into_td3(model, demo_path):
    """
    Load raw transitions directly into TD3 replay buffer.

    :param model: TD3 model
    :param demo_path: path to transitions.npz
    """
    data = np.load(demo_path)

    observations      = data["observations"]
    next_observations = data["next_observations"]
    actions           = data["actions"]
    rewards           = data["rewards"]
    dones             = data["dones"]

    print(f"\nLoading {len(observations)} transitions into replay buffer...")

    for i in range(len(observations)):
        model.replay_buffer.add(
            obs=observations[i],
            next_obs=next_observations[i],
            action=actions[i],
            reward=np.array([rewards[i]]),
            done=np.array([dones[i]]),
            infos=[{}]
        )

    print(f"Replay buffer size: {model.replay_buffer.size()} / {model.replay_buffer.buffer_size}")
    return model


# ─────────────────────────────────────────
# Run keyboard demo
# ─────────────────────────────────────────
transitions = keyboard_demo_to_buffer(demo_dir="demos/")

# ─────────────────────────────────────────
# Load into TD3 and train
# ─────────────────────────────────────────
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import time

log_dir = "model/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("LunarLanderContinuous-v3")
env = Monitor(env, log_dir)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

model = TD3(
    'MlpPolicy', env,
    action_noise=action_noise,
    verbose=1,
    device="auto",
    gradient_steps=3,
    learning_starts=100
)

# Prefill buffer with human transitions
model = load_transitions_into_td3(model, "demos/transitions.npz")

# # Train from human knowledge
# start_wall = time.perf_counter()
# model.learn(total_timesteps=int(100000), callback=callback)
# end_wall = time.perf_counter()

# print(f"Training time: {end_wall - start_wall:.2f} seconds")
# model.save(os.path.join(log_dir, "td3_from_human_demos"))
# print("Training complete!")

# # Plot learning curve
# import matplotlib.pyplot as plt

# x, y = ts2xy(load_results(log_dir), "timesteps")
# x = x[len(x) - len(y):]

# fig = plt.figure('Learning Curve')
# plt.plot(x, y)
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning Curve")
# plt.show()