import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.ppo2.ppo2 import safemean
from collections import deque

from tensorflow import losses

import numpy as np

# Tuned
from baby.rl.runner import Runner
from baby.rl.policies import build_policy


class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model
        train():
        - Make the training part (feedforward and retropropagation of gradients)
        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            st_coef=1.0, pg_coef=1.0, pg_lt_coef=0.1):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps
        
        self.start_ent_coef = ent_coef
        self.start_vf_coef = vf_coef
        self.start_st_coef = st_coef
        self.start_pg_coef = pg_coef
        self.start_pg_lt_coef = pg_lt_coef
        self.max_grad_norm = max_grad_norm


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        ADV_LT = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        R_LT = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        SR = tf.placeholder(tf.float32, [nbatch])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)
        
        # Short-term loss        
        st_loss = tf.reduce_mean(SR * neglogpac)
        
        # Long-term
        vf_loss_lt = losses.mean_squared_error(tf.squeeze(train_model.vf_lt), R_LT)        
        pg_loss_lt = tf.reduce_mean(ADV_LT * neglogpac)

        # Loss:
        # PG loss full-term + long-term
        # Value loss: full-term + long-term
        # Entropy (as usual)
        # Short-term loss: direct reward
        loss = pg_coef*pg_loss - entropy*ent_coef + vf_loss * vf_coef + st_coef*st_loss + pg_lt_coef * pg_loss_lt + vf_coef * vf_loss_lt

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        # Test with tf.train.AdamOptimizer(learning_rate=LR) doesn't show improvement

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def update_loss_trainer(progress):
            loss = pg_coef*pg_loss - entropy*ent_coef + vf_loss * vf_coef + st_coef*st_loss + pg_lt_coef * pg_loss_lt + vf_coef * vf_loss_lt

            # Update parameters using loss
            # 1. Get the model parameters
            params = find_trainable_variables("a2c_model")

            # 2. Calculate the gradients
            grads = tf.gradients(loss, params)
            if self.max_grad_norm is not None:
                # Clip the gradients (normalize)
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads = list(zip(grads, params))
            
            _train = trainer.apply_gradients(grads)
            

        def train(obs, states, rewards, masks, actions, values, values_lt, rewards_st, rewards_lt):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            advs_lt = rewards_lt - values_lt
            
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, SR:rewards_st, ADV_LT:advs_lt, R_LT:rewards_lt}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, short_term_loss, value_lt_loss, policy_lt_loss, _ = sess.run(
                [pg_loss, vf_loss, entropy, st_loss, vf_loss_lt, pg_loss_lt, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, short_term_loss, value_lt_loss, policy_lt_loss


        self.train = train
        self.update_loss_trainer = update_loss_trainer
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.value_lt = step_model.value_lt
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)

def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    model_save_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''



    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, values_lt, epinfos, rewards_st, rewards_lt = runner.run()
        epinfobuf.extend(epinfos)

        policy_loss, value_loss, policy_entropy, st_loss, v_lt_loss, pg_lt_loss = model.train(obs, states, rewards, masks, actions, values, values_lt, rewards_st, rewards_lt)
        nseconds = time.time()-tstart
        
        # Compute training progress
        progress = update*nbatch / total_timesteps # Progression from 0.0 to 1.0
        
        # Update loss trainer
        model.update_loss_trainer(progress)
        
        # Code for Automatic Domain Randomization ADR
        # Update baby_env parameter with complexity progression
        complexities = runner.adr(progress)

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("short_term_loss", float(st_loss))
            logger.record_tabular("value_lt_loss", float(v_lt_loss))
            logger.record_tabular("pg_lt_loss", float(pg_lt_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            
            logger.record_tabular("complexity", np.mean(complexities))
            logger.dump_tabular()

            if model_save_path is not None:
                model.save(model_save_path)
    return model

