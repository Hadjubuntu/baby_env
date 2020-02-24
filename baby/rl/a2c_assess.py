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

Average episode length=173.23
Std episode length=131.5873743943544
Average episode duration=0.3883528232574463 seconds

"""

import gym
import numpy as np
import time
import pathlib
from baselines.common.policies import build_policy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import baby
import baby.envs.baby_env as benv
from baby.rl.a2c import Model
from baby.rl.a2c_xp import conv_net, make_env

model_path='/home/adrien.hadj/model_baby.pkl'
network = conv_net([(64, 1, 1), (32, 1, 1)])

def load_model(venv, load_path, network, **network_kwargs):
    policy = build_policy(venv, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=venv, nsteps=0, ent_coef=0, vf_coef=0,
        max_grad_norm=0, lr=0, alpha=0, epsilon=0, total_timesteps=0, lrschedule='linear')
    
    model.load(load_path)

    return model

# Force a random seed
benv.default_conf['seed']=np.random.randint(99999)

# Create environment and gredy heuristic
venv = DummyVecEnv([make_env('baby-v0') for _ in range(1)])
env = venv.envs[0]
model = load_model(venv, model_path, network)

# Execute one episode
def episode():
    done = False
    steps = 0
    obs = env.reset()

    while not done:
        action = model.step(obs)[0]
        obs, rew, done, info = env.step(action)
        steps += 1

    return steps

# Batch execution to estimate average completion
n_runs=100
t_s = time.time()
steps=[episode() for _ in range(n_runs)]
dt = time.time()-t_s

print(f"Average episode length={np.mean(steps)}")
print(f"Std episode length={np.std(steps)}")
print(f"Average episode duration={dt/n_runs} seconds")
