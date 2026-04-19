import numpy as np
import matplotlib.pyplot as plt

def train(env, agent, episodes=10_000, max_steps=200):
    """ Trains the given agent in the provided environment for a specified number of episodes.

    Args:
        env (_type_): The environment in which the agent will be trained.
        agent (_type_): The agent to be trained.
        episodes (_type_, optional): The number of episodes to train for. Defaults to 10_000.
        max_steps (int, optional): The maximum number of steps per episode. Defaults to 200.

    Returns:
        _type_: A tuple containing the reward history and success history.
    """
    reward_history = []
    success_history = []

    print(f"Starting training for {episodes} episodes...")

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        success = False

        for _ in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

            if done:
                success = (reward >= 5.0) 
                break

        agent.decay()
        
        reward_history.append(total_reward)
        success_history.append(1 if success else 0)

        if (ep + 1) % 1000 == 0:
            recent_success_rate = np.mean(success_history[-100:]) * 100
            print(f"  Ep {ep + 1:5d} | Success Rate: {recent_success_rate:5.1f}% | Epsilon: {agent.eps:.4f}")

    return reward_history, success_history

def plot_training(reward_history, success_history, window=100):
    """ Plots the training curves for reward and success rate.

    Args:
        reward_history (_type_): A list of rewards obtained during training.
        success_history (_type_): A list of success indicators (0 or 1) during training.
        window (int, optional): The window size for the moving average. Defaults to 100.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training Curves — U-Boot V2", fontsize=12, fontweight="bold")

    success_smooth = np.convolve(success_history, np.ones(window) / window, "valid")
    reward_smooth = np.convolve(reward_history, np.ones(window) / window, "valid")

    plots_config = [
        {
            "data": success_smooth * 100,
            "color": "#1565C0",
            "title": "Success Rate",
            "ylabel": "Success (%)",
            "hline": 75,
            "ylim": (0, 105),
            "legend": ["Moving Average", "Target 75%"]
        },
        {
            "data": reward_smooth,
            "color": "#2E7D32",
            "title": "Average Reward",
            "ylabel": "Reward",
            "hline": 0,
            "ylim": None, # Auto-scale
            "legend": ["Moving Average", "Zero Line"]
        }
    ]

    for ax, config in zip(axes, plots_config):
        x_vals = range(len(config["data"]))
        
        ax.plot(x_vals, config["data"], color=config["color"], lw=1.5, label=config["legend"][0])
        
        ax.fill_between(x_vals, config["data"], alpha=0.2, color=config["color"])
        
        ax.axhline(config["hline"], color="red", ls="--", lw=1, label=config["legend"][1])
        
        ax.set(title=config["title"], xlabel="Episode (Moving Avg)", ylabel=config["ylabel"])
        ax.grid(True, alpha=0.3)
        
        if config["ylim"]:
            ax.set_ylim(config["ylim"])
            
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])     
    import os
    os.makedirs("results", exist_ok=True)
    
    plt.savefig("results/v2_curves.png", dpi=150, bbox_inches="tight")
    print("Plot saved to 'results/v2_curves.png'")
    plt.show()