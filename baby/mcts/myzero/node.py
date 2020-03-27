import numpy as np

import cloudpickle
import pickle

class Node():    
    def __init__(self, env_state):
        self.visit_count = 0
        self.reward = 0.0
        self.cum_rewards = 0.0

        # Dict for childre action to child Node
        self.children = {}
        self.infos = {}

        self.done = (env_state.validation == 0.0).all()
        self.state = {
            't': env_state.t,
            'validation_frame': np.copy(env_state.validation),
            'cum_rew': env_state.cum_rew,
        }

    def terminal(self):
        return self.done

    def get_state(self, env_state):
        # Restore environment state
        env_state.t = self.state['t']
        env_state.validation = np.copy(self.state['validation_frame'])
        env_state.cum_rew = self.state['cum_rew']

        # Restore also current observation in environment state
        obs = env_state.predict(env_state.t)
        obs = np.append(obs, env_state.validation, axis=2)
        env_state.current_obs = obs

        return env_state

    def expanded(self):
        return (len(self.children) > 0)

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.cum_rewards / self.visit_count
