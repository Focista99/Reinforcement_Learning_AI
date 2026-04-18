class RandomAgent:
    """A simple agent that selects actions randomly from the action space.
    """
    def __init__(self,action_space):
        self.action_space = action_space

    def select_action(self, observation):
        return self.action_space.sample()