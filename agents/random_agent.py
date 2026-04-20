class RandomAgent:
    """A simple agent that selects actions randomly from the action space.
    """
    def __init__(self,action_space):
        self.action_space = action_space

    def select_action(self, observation):
        """Selects an action randomly from the action space.

        Args:
            observation (_type_): The current observation from the environment (not used in this agent). 

        Returns:
            _type_: A randomly selected action from the action space.
        """
        return self.action_space.sample()