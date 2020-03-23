from baby.mcts.myzero.myzero import MyZero
import gym
import numpy as np
import time

import baby
import baby.envs.baby_env as benv

# Create environment and gredy heuristic
env = gym.make('baby-v0')

mcts = MyZero(env)
mcts.run()