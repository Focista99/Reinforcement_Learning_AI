import matplotlib.pyplot as plt

from env.uboat_env import make_env
from env.wrappers import Monitor
from agents.q_agent import Agent
from utils.training import train, plot_training
from utils.animation import run_animation

if __name__ == "__main__":
    print("=" * 50)
    print("  U-BOOT V2")
    print("=" * 50)

    env = Monitor(make_env())
    agent = Agent()

    print("\nTraining (10,000 episodes)...")

    reward_history, success_history = train(env, agent)
    env.print_stats()
    env.close()

    plot_training(reward_history, success_history)
    plt.close("all")

    run_animation(agent)