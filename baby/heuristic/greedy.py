import numpy as np
import logging

class GreedyHeuristic():
    def __init__(self, n=1):
        """
        n: Select n elements
        """
        logging.info("Greedy heuristic created")
        self.n = n

    def act(self, obs):
        # Note last frame shall be validation
        obs_h = obs[..., 0] * obs[..., -1]

        # Get action for first element
        actions = np.argsort(-obs_h.flatten())[:self.n]
        
        return actions