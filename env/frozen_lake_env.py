import gymnasium as gym
from resources.config import MAP_NAME, IS_SLIPPERY, RENDER_MODE

def create_env():
    """Creates a Frozen Lake environment.

    Returns:
        gym.Env: The created Frozen Lake environment.
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        render_mode=RENDER_MODE
    )
    return env