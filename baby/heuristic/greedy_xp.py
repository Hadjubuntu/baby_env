"""
This script aims to evaluate greedy heuristic with baby env

Results with following configuration

default_conf = {
    # Seed for data generation
    'seed': 0,
    # Number of frames
    'n_frame': 5,
    # n-xaxis (mock longitude)
    'n-xaxis': 4,
    # n-yaxis (mock latitude)
    'n-yaxis': 4,
    # max episode iteration
    'max_episode_iteration': 1000,
    # factor of ground truth modification
    'alpha_ground_truth': 0.8,
    # Minimum value of ground truth to be validated
    'validation_threshold': 0.3,
    # Sigma of gaussian filter for prediction depending on delta time
    'sigma_prediction': 0.1
}

Average episode length=157.736
Std episode length=111.51488826161285
Average episode duration=0.20047351908683778 seconds
"""

import gym
import numpy as np
import time

import baby
from baby.heuristic.greedy import GreedyHeuristic
import baby.envs.baby_env as benv

# Force a random seed
benv.default_conf['seed']=np.random.randint(99999)

# Create environment and gredy heuristic
env = gym.make('baby-v0')
heuristic = GreedyHeuristic(n=1)

# Execute one episode
def episode():
    done = False
    steps = 0
    obs = env.reset()

    while not done:
        action = heuristic.act(obs)[0]
        obs, rew, done, info = env.step(action)

        steps += 1

    return steps

# Batch execuion to estimate average completion
n_runs=1000
t_s = time.time()
steps=[episode() for _ in range(n_runs)]
dt = time.time()-t_s

print(f"Average episode length={np.mean(steps)}")
print(f"Std episode length={np.std(steps)}")
print(f"Average episode duration={dt/n_runs} seconds")
