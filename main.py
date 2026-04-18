import time 
from env.frozen_lake_env import create_env
from agents.random_agent import RandomAgent
from resources.config import PAUSE_BETWEEN_STEPS

def execute_mission_u_boat():
    """ Executes the mission for the U-Boat agent in the Frozen Lake environment.
        The agent will take random actions until it either reaches the goal or falls into a hole. 
        The environment is rendered at each step, and the outcome of the mission is printed at the end.
    """
    env = create_env()
    agent = RandomAgent(env.action_space)

    observation, info = env.reset()
    done = False
    truncated = False
    steps = 0

    while not (done or truncated):
        action = agent.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        env.render()
        time.sleep(PAUSE_BETWEEN_STEPS)

    if reward == 1:
        print(f"Mission accomplished in {steps} steps!")
    else:
        print(f"Mission failed after {steps} steps.")
    env.close()

if __name__ == "__main__":
    execute_mission_u_boat()

