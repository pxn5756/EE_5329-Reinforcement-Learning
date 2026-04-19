# Continuous Control of Lunar Lander

# Installation

## Requirements:

Python: Between Python 3.11 and 3.13  
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
   
# Training, Evaluation and Comparison:

## Training:

Command: python train.py

Training is executed by running the train.py file.
You may edit this file as needed to modify the model hyperparameters
or number of training steps. The model and environment are defined
in the main function. A call to the "train" function is made
to perform the learning. This function will also handle plotting
and saving of the results. The two callback classes in this file are
for diagnostic purposes and provide feedback on the performance
of the model as it is learning. 

The final model and all artifacts are saved to the corresponding
"model/experiment_name" folder.

Note: A  seed of 42 is used during the training

The load_results.py file is a simple script for quickly plotting
previously saved training results for closer analysis of the learning
curve.

## Evaluation:

There are two scripts available for evaluation.
Note: A seed of 1 is used for evaluation.

### eval.py

The eval.py file allows for a model to be evaluated over 100 episodes.
The reward evaluation curve, mean rewards, and standard deviation will
be plotted and saved to the respective model folder.

### eval_anime.py

The eval_anime.py runs a small evaluation of 10 episodes. The evaluation will
be rendered in a small window for the user to physically see the Lunar Lander
behavior. When executed a window will appear prompting the
user to select a model to evaluate.

## Comparison

The compare_models.py script is used to provide charts for comparing all models
that have been trained. The script will iterate and obtain the training and 
evaluation logs and plot them on a single plot for comparing. Additionally,
a bar graph is provided for examining mean and standard deviation across
the models.

