import random
from collections import defaultdict
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper, ActionWrapper, Wrapper
from gymnasium.spaces import Box

class ObsWrapper(ObservationWrapper):
    """
    Observation wrapper that transforms the agent's discrete state into a
    continuous vector of 4 normalized features.

    The observation vector contains:
        - Normalized row of the agent (0.0 to 1.0)
        - Normalized column of the agent (0.0 to 1.0)
        - Normalized distance to the goal
        - Normalized distance to the nearest mine

    Args:
        env: Gymnasium environment to wrap.
        grid_map (list | np.ndarray, optional): Grid map. If not provided,
            it is obtained from the ``desc`` attribute of the base environment.

    Raises:
        ValueError: If the environment has no ``desc`` attribute and no
            ``grid_map`` is provided, or if no goal cell ('G') is found in the map.
    """

    def __init__(self, env, grid_map=None):
        super().__init__(env)
        
        if grid_map is not None:
            self.grid_map = grid_map
        else:
            base_env = self.env.unwrapped
            if not hasattr(base_env, 'desc'):
                raise ValueError("ObsWrapper requires an environment with a 'desc' attribute or an explicit grid_map.")
            self.grid_map = base_env.desc
            
        if isinstance(self.grid_map, list):
            self.g = len(self.grid_map)
        else:
            self.g = self.grid_map.shape[0]
        
        self._mines = []
        self._goal = None
        
        for r in range(self.g):
            for c in range(self.g):
                if isinstance(self.grid_map, list):
                    cell = self.grid_map[r][c]
                else:
                    cell = self.grid_map[r, c] if isinstance(self.grid_map, np.ndarray) else self.grid_map[r][c]

                if isinstance(cell, bytes):
                    cell = cell.decode("utf-8")
                elif isinstance(cell, np.str_):
                    cell = str(cell)
                    
                if cell == 'H':
                    self._mines.append((r, c))
                elif cell == 'G':
                    self._goal = (r, c)
        
        if self._goal is None:
            raise ValueError("No Goal ('G') found in the map.")
            
        self.observation_space = Box(
            low=np.zeros(4, dtype=np.float32), 
            high=np.ones(4, dtype=np.float32), 
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Transforms the discrete observation (cell index) into a continuous vector.

        Args:
            obs (int): Linear index of the agent's current cell.

        Returns:
            np.ndarray: Shape-(4,) array with normalized features
                ``[row, col, dist_goal, dist_mine]``, all in [0.0, 1.0].
        """
        r, c = divmod(obs, self.g)
        gr, gc = self._goal
        
        norm_r = r / (self.g - 1) if self.g > 1 else 0.0
        norm_c = c / (self.g - 1) if self.g > 1 else 0.0
        
        dist_goal = np.hypot(r - gr, c - gc)
        max_dist = np.sqrt(2) * (self.g - 1)
        norm_dist_goal = dist_goal / max_dist if max_dist > 0 else 0.0
        
        if self._mines:
            dist_mine = min(np.hypot(r - mr, c - mc) for mr, mc in self._mines)
            norm_dist_mine = min(1.0, dist_mine / (self.g / 2))
        else:
            norm_dist_mine = 1.0  
            
        return np.array([norm_r, norm_c, norm_dist_goal, norm_dist_mine], dtype=np.float32)

class RewWrapper(RewardWrapper):
    """
    Reward wrapper that implements a 'Reward Shaping' technique to provide 
    denser feedback to the agent during training.
    
    Instead of a sparse reward, it calculates the progress made towards the 
    goal based on the distance provided by the observation vector.
    """

    def __init__(self, env):
        super().__init__(env)
        # Store the distance from the previous step to calculate progress
        self.prev_dist = None

    def reset(self, **kwargs):
        """
        Resets the environment and initializes the starting distance to goal.
        
        Returns:
            tuple: (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        
        # According to ObsWrapper, index 2 is 'dist_goal_norm'
        self.prev_dist = obs[2]
        return obs, info

    def step(self, action):
        """
        Executes one environment step and calculates the reward based on 
        terminal states and distance-based progress.
        
        Args:
            action (int): The action performed by the agent.
            
        Returns:
            tuple: (obs, new_reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Extract current distance to goal from the observation vector
        curr_dist = obs[2]

        # 2. Calculate progress (positive if closer, negative if further)
        # progress > 0 means the distance decreased
        progress = self.prev_dist - curr_dist
        
        # 3. Update previous distance for the next step calculation
        self.prev_dist = curr_dist

        # 4. Determine final reward value
        if info.get("fuel_depleted"):
            # Penalty for running out of fuel
            new_reward = -2.0
        elif terminated:
            if reward == 1.0:
                # Big bonus for reaching the convoy
                new_reward = 10.0
            else:
                # Penalty for hitting a mine
                new_reward = -5.0
        else:
            # Dense reward: base step penalty + weighted progress bonus
            # We use a multiplier (2.0) to make the signal more significant
            new_reward = -0.01 + (progress * 2.0)
            
        return obs, new_reward, terminated, truncated, info
    
    
class FuelWrapper(Wrapper):
    """
    Wrapper that introduces a fuel constraint for the agent.

    When fuel drops below 20% of its maximum value, upward (0) or leftward (3)
    movement actions are randomly redirected to downward (1) or rightward (2),
    simulating an emergency maneuver.

    If fuel reaches zero before the episode ends, the episode is truncated and
    a penalty of ``-2.0`` is applied.

    Args:
        env: Gymnasium environment to wrap.
        fuel (int): Initial fuel amount per episode. Defaults to 150.
    """

    def __init__(self, env, fuel=150):
        super().__init__(env)
        self.max_fuel = fuel
        self.fuel = fuel

    def _maybe_redirect(self, action):
        """Redirects action when fuel is critically low."""
        if self.fuel <= self.max_fuel * 0.2 and action in (0, 3):
            return random.choice([1, 2])
        return action

    def reset(self, **kwargs):
        """
        Resets the environment and restores fuel to its maximum value.

        Args:
            **kwargs: Additional arguments passed to the underlying
                environment's ``reset`` method.

        Returns:
            tuple: ``(obs, info)`` returned by the underlying environment.
        """
        self.fuel = self.max_fuel
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        """
        Executes one step, decrements fuel, applies the critical-fuel redirect,
        and truncates the episode if fuel is exhausted.

        Args:
            action (int): Action selected by the policy.

        Returns:
            tuple: ``(obs, reward, terminated, truncated, info)`` with the keys
                ``fuel_depleted`` (bool) and ``fuel_remaining`` (int) added to
                ``info``. If fuel is exhausted, ``truncated`` is set to ``True``
                and the reward is ``-2.0``.
        """
        self.fuel -= 1

        # Apply redirection directly — no double-transform bug possible here
        action = self._maybe_redirect(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.fuel <= 0 and not (terminated or truncated):
            truncated = True
            reward = -2.0
            info["fuel_depleted"] = True
        else:
            info["fuel_depleted"] = False

        info["fuel_remaining"] = self.fuel
        return obs, reward, terminated, truncated, info


class Monitor(Wrapper):
    """
    Monitoring wrapper that records episode statistics.

    Classifies each finished episode into one of four categories:
        - ``"ok"``:      The agent successfully reached the goal.
        - ``"mine"``:    The agent collided with a mine.
        - ``"fuel"``:    The agent ran out of fuel.
        - ``"timeout"``: The episode was truncated for another reason (time limit).

    Attributes:
        stats (defaultdict[str, int]): Episode count per category.
        steps (list[int]): Number of steps in each completed episode.
        current_steps (int): Step counter for the episode currently in progress.
    """

    def __init__(self, env):
        super().__init__(env)
        self.stats = defaultdict(int)
        self.steps = []
        self.current_steps = 0

    def reset(self, **kwargs):
        """
        Resets the environment and the step counter for the current episode.

        Args:
            **kwargs: Additional arguments passed to the underlying
                environment's ``reset`` method.

        Returns:
            tuple: ``(obs, info)`` returned by the underlying environment.
        """
        self.current_steps = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        """
        Executes one step, increments the step counter, and — when the episode
        ends — records its length and classifies it in ``stats``.

        Args:
            action: Action to execute in the underlying environment.

        Returns:
            tuple: ``(obs, reward, terminated, truncated, info)`` unchanged
                from the underlying environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps += 1
        
        if terminated or truncated:
            self.steps.append(self.current_steps)
            
            if info.get("fuel_depleted"):
                key = "fuel"
            elif terminated and reward >= 5.0: 
                key = "ok"
            elif terminated:
                key = "mine"
            else:
                key = "timeout"
                
            self.stats[key] += 1
            
        return obs, reward, terminated, truncated, info

    def print_stats(self):
        """
        Prints a summary of the accumulated statistics to the console.

        Displays the count and percentage of episodes per category, as well as
        the average number of steps per episode. If no episodes have been
        recorded yet, a notice is printed and the method returns early.
        """
        total = sum(self.stats.values())
        if total == 0:
            print("\n=== STATISTICS ===\nNo episodes recorded yet.")
            return

        print("\n=== STATISTICS ===")
        labels = [
            ("ok", "Convoy intercepted"),
            ("mine", "Sunk by mine"),
            ("fuel", "Out of fuel"),
            ("timeout", "Time out")
        ]
        
        for key, label in labels:
            count = self.stats[key]
            percentage = (count / total) * 100
            print(f"  {label:22s}: {count:4d}  ({percentage:5.1f}%)")
            
        if self.steps:
            print(f"  Average steps: {np.mean(self.steps):.1f}")
