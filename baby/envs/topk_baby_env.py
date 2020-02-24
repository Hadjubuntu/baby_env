"""
Wrapper of baby env
"""
from baby.envs.baby_env import BabyEnv
from baby.heuristic.greedy import GreedyHeuristic
from gym import spaces


class TopkBabyEnv(BabyEnv):
    def __init__(self):
        super().__init__()

        self.k = 2
        self.action_space = spaces.Discrete(self.k)

        self.f = GreedyHeuristic(n=self.k)

    def step(self, action):
        # Heuristic action
        action_f = self.f.act(self.current_obs)
        action_exec = action_f[action]

        obs, rew, done, info = super().step(action_exec)

        return obs, rew, done, info




