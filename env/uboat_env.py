import gymnasium as gym
from env.map_generator import generate_map


def make_base_env(seed=None):
    """
    Creates a bare FrozenLake-v1 environment using a procedurally generated map.

    Generates an 8x8 U-Boot grid map with 15% mine density and wraps it
    inside a standard Gymnasium FrozenLake environment with no slipperiness
    and no render mode. The TimeLimit wrapper is removed to allow the
    FuelWrapper to control episode termination based on fuel depletion.

    Args:
        seed (int, optional): Random seed passed to the map generator for
            reproducibility. If ``None``, the map is generated randomly.

    Returns:
        gymnasium.Env: Unwrapped FrozenLake-v1 environment configured with
            the generated map, without TimeLimit wrapper.
    """
    grid_size = 8
    uboat_map = generate_map(grid=grid_size, density=0.15, seed=seed)

    env = gym.make(
        "FrozenLake-v1",
        desc=uboat_map,
        is_slippery=False,
        render_mode=None,
        max_episode_steps=None  # Remove time limit to let fuel mechanic control episode length
    )

    return env


def make_env(seed=None):
    """
    Creates a fully wrapped U-Boot environment ready for agent training.

    Builds on top of ``make_base_env`` by applying three wrappers in order:

    1. ``ObsWrapper``  — transforms the discrete cell index into a
    continuous 4-feature observation vector.
    2. ``FuelWrapper`` — introduces a fuel constraint that truncates the
    episode when fuel is exhausted.
    3. ``RewWrapper``  — replaces the sparse binary reward with a denser
    reward scheme.

    The same procedurally generated map is shared between the base
    environment and ``ObsWrapper`` to guarantee consistency.

    Args:
        seed (int, optional): Random seed passed to the map generator for
            reproducibility. If ``None``, the map is generated randomly.

    Returns:
        gymnasium.Env: Fully wrapped environment exposing a continuous
            observation space and the custom reward and fuel mechanics.
    """
    from env.wrappers import ObsWrapper, RewWrapper, FuelWrapper

    grid_size = 8
    uboat_map = generate_map(grid=grid_size, density=0.15, seed=seed)

    env = make_base_env(seed=seed)

    env = ObsWrapper(env, grid_map=uboat_map)
    env = FuelWrapper(env)
    env = RewWrapper(env)

    return env