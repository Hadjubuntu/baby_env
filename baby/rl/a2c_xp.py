
import gym

import numpy as np
import tensorflow as tf
import datetime
import time
import pathlib
import json

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import logger

import tensorflow as tf
import tensorflow.contrib.layers as layers

import baby
from baby.rl.a2c import learn


def make_env(env_name, seed=0):
    def build_env():
        env = gym.make(env_name)
        # env.seed(seed)
        return env
    return lambda: build_env()


def conv_net(convs):
    def network_fn(X):
        out = X
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu,
                                           )

        return out, None

    return network_fn

def dense(hiddens):
    def network_fn(X):
        out = X
        with tf.variable_scope("dense"):
            for nb_hidden, _, _ in hiddens:
                out = layers.fully_connected(out,
                                           num_outputs=nb_hidden,
                                           activation_fn=tf.nn.relu,
                                           )

        return out, None

    return network_fn


def kill_prev_network():
    tf.reset_default_graph()
    sess = tf.get_default_session()
    if sess:
        sess.close()


def xp(
    env_name='baby-v0',
    nenv=5,
    model_type='conv',
    network_archi=[(32, 1, 1)],
    lr=7e-4,
    nsteps=5,
    vf_coef=0.5,
    ent_coef=0.01,
    gamma=0.99,
):
    kill_prev_network()

    dir_xp = 'baby_2x2'

    d=datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H-%M-%S')
    simu_name=f"env{env_name}_lr{lr}_nsteps{nsteps}_{model_type}_{network_archi[0][0]}filters_{len(network_archi)}layers_ent{ent_coef}_gamma{gamma}"
    work_dir=f'{pathlib.Path.home()}/{dir_xp}/{simu_name}/run-{d}' 

    logger.configure(
        dir=f'{work_dir}/openai_logs',
        format_strs=['stdout', 'tensorboard'])

    model_path=f'{work_dir}/model.pkl'
    conf_path=f'{work_dir}/conf.json'
    
    venv = SubprocVecEnv([make_env(env_name) for _ in range(nenv)])

    # Dump conf to run directory
    with open(conf_path, 'w') as f:
        from baby.envs.baby_env import default_conf
        json.dump(default_conf, f, sort_keys=True, indent=4)

    # Create model
    if model_type == 'conv':
        network=conv_net(convs=network_archi)
    else:
        network=dense(hiddens=network_archi)

    # Train model    
    learn(
        network,
        venv,      
        lr=lr,
        lrschedule='linear',
        nsteps=nsteps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=0.5,
        gamma=gamma,
        total_timesteps=int(50e6),
        model_save_path=model_path
    )
    
if __name__ == '__main__':
    #xp(convs=[(32, 1, 1)], lr=7e-4, nsteps=5)
    # xp(convs=[(32, 1, 1)], lr=7e-4, nsteps=128)
    
    # test lr impact / nsteps
    xp(
        env_name='baby-v0', 
        model_type='conv', 
        network_archi=[(16, 2, 1), (8, 1, 1)],
        lr=1e-3, 
        nsteps=5, 
        vf_coef=0.5,
        ent_coef=0.001, 
        gamma=0.9)
    # xp(convs=[(32, 1, 1)], lr=1e-4, nsteps=128)
    
    # test network    
    # xp(convs=[(128, 10, 1), (128, 1, 1)], lr=1e-3, nsteps=5)
    # xp(convs=[(128, 10, 1), (128, 1, 1)], lr=7e-4, nsteps=128)
    
    # test network deep
    # xp(convs=[(128, 10, 1), (64, 1, 1), (32, 1, 1)], lr=7e-4, nsteps=5)
    # xp(convs=[(128, 10, 1), (64, 1, 1), (32, 1, 1)], lr=7e-4, nsteps=128)
    