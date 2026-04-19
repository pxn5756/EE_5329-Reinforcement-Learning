import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Open file dialog to select model
root = tk.Tk()
root.withdraw()  # hides the empty tkinter window

log_dir = filedialog.askdirectory(
    title="Select folder with monitor.csv file",
    initialdir="model/"
)

results = load_results(log_dir)
x, y = ts2xy(results, "timesteps")

# Plot
fig = plt.figure('Rewards')
plt.plot(x, y, label="Learning Curve")

plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning Curve")
plt.show()