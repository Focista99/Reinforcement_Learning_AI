import random 
from collections import defaultdict
import numpy as np 

class Agent: 
    def __init__(self, bins=8, actions=4, lr=0.1, gamma=0.99, 
                eps=1.0, eps_min=0.01, eps_decay=0.998):
        """Initializes the Q-Learning agent.

        Args:
            bins (int, optional): Number of bins to discretize the state space. Defaults to 8.
            actions (int, optional): Number of possible actions. Defaults to 4.
            lr (float, optional): Learning rate. Defaults to 0.1.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            eps (float, optional): Initial exploration rate. Defaults to 1.0.
            eps_min (float, optional): Minimum exploration rate. Defaults to 0.01.
            eps_decay (float, optional): Exploration decay factor. Defaults to 0.998.
        """
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        
        self.cuts = np.linspace(0, 1, bins + 1)[1:-1]
        self.Q = defaultdict(lambda: np.zeros(actions))

    def key(self, o):
        """Converts a continuous state into a discrete key for the Q dictionary.

        Args:
            o (tuple): Current state.

        Returns:
            tuple: Key for the Q dictionary.
        """
        return tuple(int(np.digitize(v, self.cuts)) for v in o)

    def act(self, o):
        """Selects an action using an epsilon-greedy policy.

        Args:
            o (tuple): Current state.

        Returns:
            int: Action to take.
        """
        if random.random() < self.eps:
            return random.randint(0, self.actions - 1)
        else:
            return int(np.argmax(self.Q[self.key(o)]))

    def update(self, o, a, r, o2, done):
        """Updates the Q-table using the Q-Learning update formula.

        Args:
            o (tuple): Current state.
            a (int): Action taken.
            r (float): Reward received.
            o2 (tuple): Next state.
            done (bool): Indicates whether the episode has ended.
        """
        k, k2 = self.key(o), self.key(o2)
        
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[k2])
            
        self.Q[k][a] += self.lr * (target - self.Q[k][a])

    def decay(self):
        """Decays the exploration rate."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)