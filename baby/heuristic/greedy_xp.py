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


--------------------
Conf for transfer learning

default_conf = {
    # Seed for data generation
    'seed': 0,
    # Number of frames
    'n_frame': 20, # 5
    # n-xaxis (mock longitude)
    'n-xaxis': 21,
    # n-yaxis (mock latitude)
    'n-yaxis': 9,
    # max episode iteration
    'max_episode_iteration': 1000,
    # factor of ground truth modification
    'alpha_ground_truth': 0.8,
    # Minimum value of ground truth to be validated
    'validation_threshold': 0.8,
    # Sigma of gaussian filter for prediction depending on delta time
    'sigma_prediction': 1.0, # prev=0.1
    # Reward system
    'reward': {
        # Reward at each step
        'timestep': 0,
        # Reward for each validated element
        'validation': 1,
    }
}

Average episode length=223.99
Std episode length=21.62521444980373
Average episode duration=1.2746098566055297 seconds

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

# Print frequence
print_freq = 25

# Execute one episode
def episode(i):
    if i % print_freq == 0:
        print(f"Running episode #{i}")
    done = False
    steps = 0
    obs = env.reset()

    while not done:
        action = heuristic.act(obs)[0]
        obs, rew, done, info = env.step(action)

        steps += 1

    return steps

# Batch execuion to estimate average completion
n_runs=100
t_s = time.time()
steps=[episode(i) for i in range(n_runs)]
dt = time.time()-t_s

print(f"Average episode length={np.mean(steps)}")
print(f"Std episode length={np.std(steps)}")
print(f"Average episode duration={dt/n_runs} seconds")
