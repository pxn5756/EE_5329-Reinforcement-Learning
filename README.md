# Continuous Control of Lunar Lander

# GIT Repository:
All code discussed in this readme can be found here:
https://github.com/pxn5756/EE_5329-Reinforcement-Learning.git

# Installation

## Requirements:

Python: Python 3.13.13  
Gymnasium: V1.2.3  
Stable Baselines 3: 2.8.0  

## Installation Instructions:

1. Install Python.
2. Using command prompt or bash execute the create_environment.sh
   script. This will automatically install the necesssary packages.
   and create the python environment.
3. Once the setup script has completed, activate the environment by
   typing: "LunarLander\scripts\activate"
   Successful activation will be indicated with a "(LunarLander)"
   appearing at the beginning of your command line.
   
NOTE:
- The environment is setup to install pytorch CUDA v13. If your GPU requires a different
  version, please modify the shell script for the appropriate version.
   
   
# Folder Structure:

All artifacts including the trained model and metrics
from evaluation and training are stored in the "model" folder.

Example: "model/experiment1"

This will contain subdirectories corresponding to the appropriate experiment.
The baseline mode is also provided here in the "model/baseline" folder.
Inside the respective "experiment#" folder are any logs and plots of
training/evaluation curves. The trained model for the experiment will
have the same name (i.e. "experiment1_model.zip" for example).
User recordings of the trained agent are also stored in this same folder.
The secondary eval folder contains the evaluation logs. This is kept here
as the SB3 load function selects the first "*monitor.csv" file it finds
and thus would confict with the "train_monitor.csv" log.

# Training, Evaluation and Comparison:

## Training:

There are two scripts available for training.

### train.py

Command: python train.py

Primary training is executed by running the train.py file.
You may edit this file as needed to modify the model hyperparameters
or number of training steps. The model and environment are defined
in the main function. A call to the "train" function is made
to perform the learning. This function will also handle plotting
and saving of the results. The two callback classes in this file are
for diagnostic purposes and provide feedback on the performance
of the model as it is learning. 

The final model and all artifacts are saved to the corresponding
"model/experiment_name" folder ("model/experiment#/experiment#_model.zip")

Note: A  seed of 42 is used during the training

### train_demo.py

The train_demo.py script allows the user to manually control the Lunar Lander
using the keyboard.
This allows the user to try to perform "expert" demonstrations before beginning
training. The game will allow the user to perform an unlimited amount of attempts.
To end demonstration, the user can either close the pygame window or press 'Q' to quit.
Upon quitting, the data in the replay buffer from the demonstration is temporarily saved.
The model and environment are then reset to begin a fresh training sequence.
Prior to training starting, the data from the demonstration is loaded into the replay
buffer.

#### KEYS:
Main Booster:
    - 'Up Arrow' = 100% Thrust
    - 'W' = 50% Thrust
    - 'E' = 10% Thrust
    
Side Boosters:
    - 'Left' or 'A'= Left Thrust 50%
    ' 'Right' Arrow  or 'D' = Right Thrust 50%

### load_results.py

Command: python load_results.py

The load_results.py file is a simple script for quickly plotting
previously saved training results for closer analysis of the learning
curve.

## Evaluation:

There are two scripts available for evaluation.
Note: A seed of 1 is used for evaluation to test the generalization of the agent.

### eval.py

Command: python eval.py

The eval.py file allows for a model to be evaluated over 100 episodes.
The reward evaluation curve, mean rewards, and standard deviation will
be plotted and saved to the respective model folder ("model/experiment#").
The logged eval_monitor.csv is placed inside the eval folder for the model.
NOTE: It is important that the folder eval and the file eval_monitor.csv is
not modifed as the compare_models.py script relies on this name for extracting.
In addition, the csv file must be kept in a seperate folder from the
train_monitor.csv as the SB3 load function is unable to select
and load specific monitor files and instead relies on a "*monitor.csv" file
existing in the given path.

### eval_anime.py

Command: python eval_anime.py

The eval_anime.py runs a small evaluation of 10 episodes. The evaluation will
be rendered in a small window for the user to physically see the Lunar Lander
behavior. When executed a window will appear prompting the
user to select a model to evaluate. The user must manually perform a recording.
The window is populated initially to allow the user to prepare a screen recording.


## Comparison

Command: python compare_models.py

The compare_models.py script is used to provide charts for comparing all models
that have been trained. The script will iterate and obtain the training and 
evaluation logs and plot them on a single plot for comparing. Additionally,
a bar graph is provided for examining mean and standard deviation across
the models. All plots are saved to the primary folder of the repository.

