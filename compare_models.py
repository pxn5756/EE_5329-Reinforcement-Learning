import os
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3.common.results_plotter import load_results, ts2xy

def main():
    log_dir = "model"

    subdirs = [entry.path for entry in os.scandir(log_dir) if entry.is_dir()]
    
    fig1 = plt.figure('Learning Curves')
    fig2 = plt.figure('Evaluation Curves')
    fig3 = plt.figure('Mean Rewards')

    mean_rewards = []
    std_dev_rewards = []
    labels = []
    for model in subdirs:
        if model not in ["model\\test_models"]:
            results_train = load_results(model)
            results_eval = load_results(model + "/eval")

            x_train, y_train = ts2xy(results_train, "timesteps")

            results_eval = load_results(model + "/eval")
            x_eval, y_eval = ts2xy(results_eval, "episodes")

            label = model.split("\\")[-1]

            plt.figure('Learning Curves')
            plt.plot(x_train, y_train, label=label)

            plt.figure('Evaluation Curves')
            plt.plot(x_eval, y_eval, label=label)

            labels.append(label)
            mean_rewards.append(np.mean(y_eval))
            std_dev_rewards.append(np.std(y_eval))

    plt.figure('Learning Curves')
    plt.legend()
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title("Learning Curve Comparison")
    plt.savefig("learning_curve_compare.png", dpi=300, bbox_inches="tight")

    plt.figure('Evaluation Curves')
    plt.legend()
    plt.xlabel("Number of Episodes")
    plt.ylabel("Rewards")
    plt.title("Evaluation Curve Comparison")
    plt.savefig("eval_curve_compare.png", dpi=300, bbox_inches="tight")

    plt.figure('Mean Rewards')
    plot = plt.bar(labels, mean_rewards, color="steelblue")
    plt.xlabel("Model")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward Comparison over 100 episodes")
    

    for bar in plot:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}", va='bottom', ha='center')

    plt.savefig("mean_reward_bar.png", dpi=300, bbox_inches="tight")

    plt.figure('Standard Deviation Rewards')
    plot2 = plt.bar(labels, std_dev_rewards, color="steelblue")
    plt.xlabel("Model")
    plt.ylabel("Standard Deviation of Reward")
    plt.title("Standard Deviation of Reward Comparison over 100 episodes")
    for bar in plot2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}", va='bottom', ha='center')
    
    plt.savefig("std_dev_reward_bar.png", dpi=300, bbox_inches="tight")



    plt.show()

if __name__ == "__main__":
    main()