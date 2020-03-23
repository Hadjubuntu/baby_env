import numpy as np

import cloudpickle
import pickle

class Node():    
    def __init__(self, env_state):
        self.visit_count = 0
        self.reward = 0.0
        self.cum_rewards = 0.0
        self.children = []

        self.state = cloudpickle.dumps(env_state)

    def get_state(self):
        return pickle.loads(self.state)

    def expanded(self):
        return (len(self.children) > 0)

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.cum_rewards / self.visit_count