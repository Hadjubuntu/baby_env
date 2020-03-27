from baby.mcts.myzero.myzero import MyZero
import gym
import numpy as np
import time

import baby
import baby.envs.baby_env as benv

# Create environment and wrap-up into MCTS algorithm
env = gym.make('baby-v0')
mcts = MyZero(env)

# Iteration of MCTS to train models
n_iteration = 100

for _ in range(n_iteration):
    # A MCTS run, reset episode, and train for n_simulation (see MCTS conf)
    mcts.run()