import gymnasium as gym
def create_env():
    """Creates a Frozen Lake environment.

    Returns:
        gym.Env: The created Frozen Lake environment.
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=None , 
        map_name="4x4",
        is_slippery=False,
        render_mode="human"
    )
    return env 

