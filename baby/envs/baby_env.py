import gym
from gym import spaces

import logging
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import uniform_filter


import numpy as np

LOGGER = logging.getLogger(__name__)

default_conf = {
    # Seed for data generation
    'seed': 0,
    # Number of frames
    'n_frame': 20, # 20
    # n-xaxis (mock longitude, default: 21)
    'n-xaxis': 21,
    # n-yaxis (mock latitude, default: 9)
    'n-yaxis': 9,
    # max episode iteration
    'max_episode_iteration': 1000,
    # factor of ground truth modification
    'alpha_ground_truth': 0.8,
    'alpha_slow_ground_truth': 0.1,
    # Number of slow frames (with slow alpha factor)
    'alpha_slow_freq': 2,
    # Minimum value of ground truth to be validated
    'validation_threshold': 0.8,
    # Sigma of gaussian filter for prediction depending on delta time
    'sigma_prediction': 1.0, # prev=1.0 for training/transfer // prev=0.1
    'sigma_gaussian_value': 0.5,
    'gamma_gaussian_value': 0.05,

    # Reward system
    'reward': {
        # Reward at each step
        'timestep': 0,
        # Reward for each validated element
        'validation': 1,
        # Reward at completion (episode done)
        'done': 0,
    },

    # Use Automatic Domain Randomization
    'ADR': False,
}

# Range for Automatic Domain Randomization
conf_adr = {
    # factor of ground truth modification
    'alpha_ground_truth': [0.6,1.4],
    # Sigma of gaussian filter for prediction depending on delta time
    'sigma_prediction': [0.1, 0.3],
    'sigma_gaussian_value': [0.05, 0.2],
    'gamma_gaussian_value': [0.0, 0.1],
}

class BabyEnv(gym.Env):
    def __init__(self):
        self.conf = default_conf
        self.np_random = np.random
        self.np_random.seed(self.conf['seed'])
        
        self.t = 0
        self.cum_rew = 0
        self.current_obs = None
        self.ground_truth = np.zeros((self.conf['n-yaxis'], self.conf['n-xaxis'], self.conf['max_episode_iteration']))
        self.validation = np.ones((self.conf['n-yaxis'], self.conf['n-xaxis']))
        
        obs_shape = (self.conf['n-yaxis'], self.conf['n-xaxis'], self.conf['n_frame']+1)
        self.action_space = spaces.Discrete(self.conf['n-yaxis'] * self.conf['n-xaxis'])
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                       shape=obs_shape,
                                       dtype=np.float32)
        
    def adr(self, progress):
        if self.conf['ADR']:
            complexity = []
            
            for conf_key in conf_adr.keys():
                param_var = conf_adr[conf_key]
                
                gamma = np.random.normal(loc=progress,scale=0.2)
                value = gamma*(param_var[1] - param_var[0]) + param_var[0]
                value = np.clip(value, param_var[0], param_var[1])
                
                # Modify real conf
                self.conf[conf_key] = value
                complexity.append(value)
                
            return np.mean(complexity)
        else:
            # Deactivated ADR solution
            return 0.0
        
    def reset(self):
        self.t = 0
        self.cum_rew = 0
        self.reset_ground_truth()
        self.validation = np.ones((self.conf['n-yaxis'], self.conf['n-xaxis']))
        self.validation = np.expand_dims(self.validation, -1)
        
        obs = self.predict(self.t)
        obs = np.append(obs, self.validation, axis=2)

        self.current_obs = obs
        
        return obs    
    
    
    def step(self, action):
        rew = 0.0
        done = False
        info = {'episode': 0.0}
        # Apply action and compute reward
        current_truth = self.ground_truth[..., self.t]
        y, x = np.unravel_index(action, dims=current_truth.shape)
        
        validation_frame = np.squeeze(self.validation)             

        if current_truth[y, x] > (1.0-self.conf['validation_threshold']) and validation_frame[y, x] == 1.0:
            rew += self.conf['reward']['validation']
            validation_frame[y, x] = 0.0
            self.validation = np.expand_dims(validation_frame, -1)
        
        rew += self.conf['reward']['timestep']
        
        # Update t
        self.t += 1
        
        # Update obs        
        obs = self.predict(self.t)
        obs = np.append(obs, self.validation, axis=2)
        self.current_obs = obs

        # Cumulative reward
        self.cum_rew += rew
        
        # Status done
        if (self.validation == 0.0).all() or self.t >= (self.conf['max_episode_iteration'] - self.conf['n_frame']):
            done = True
            
            # Reward if episode done
            rew = self.conf['reward']['done']  
            self.cum_rew += rew
            
            # Collect info
            info = {'episode': {'r': self.cum_rew, 'l': self.t}}
        
        return obs, rew, done, info
    
    def sigma_v(self, v, mu=0.5, sigma=1.2):
        return np.exp(-(v-mu)**2/(2*sigma**2))

    
    def f_predict_tuned(self, truth_frame, dt):
        """
        Tuned to imitate uncertainty behaviour depending on gd values
        """
        # Add blur depending on frame time
        sigma_t = self.conf['sigma_prediction'] * np.sqrt(dt+1)

        # Add bias depending on truth value (on median values)
        rand_frame = (self.np_random.rand(self.conf['n-yaxis'], self.conf['n-xaxis'])-0.5)
        rand_m = self.conf['gamma_gaussian_value'] * np.sqrt(dt+1) * self.sigma_v(
            truth_frame, mu=0.5, sigma=self.conf['sigma_gaussian_value']
        ) * rand_frame
        
        pred_t = gaussian_filter(truth_frame+rand_m, sigma=sigma_t)
        
        return pred_t

    def f_predict(self, truth_frame, dt):        
        """
        Origin method (simple gaussian filter with sigma depending on t)
        """   
        pred = gaussian_filter(truth_frame, sigma=self.conf['sigma_prediction']*(dt+1))
        
        return pred

    
    def predict(self, t):
        n_frame = self.conf['n_frame']
        pred_t_tn = np.copy(self.ground_truth[..., t:(t+n_frame)])
                
        for i in range(self.conf['n_frame']):
            pred_t_tn[..., i] = self.f_predict_tuned(pred_t_tn[..., i], i)            
              
        # Clip all values if outside [0, 1]    
        pred_t_tn = np.clip(pred_t_tn, 0.0, 1.0)
        
        return pred_t_tn
        
    
    def f_truth(self, prev_frame, alpha):
        """
        Apply function to previous frame to compute next frame
        """
        next_frame = np.copy(prev_frame)
        
        rand_frame = (self.np_random.rand(self.conf['n-yaxis'], self.conf['n-xaxis'])-0.5)
        next_frame = next_frame + alpha * gaussian_filter(rand_frame, sigma=1.0)        
        
        # Need to clip here to prevent from diverging recursively
        next_frame = np.clip(next_frame, 0.0, 1.0)

        return next_frame
    
    def reset_ground_truth(self):
        tmp = np.copy(self.ground_truth)

        # Generate the first frame randomly, apply gaussian filter to imitate spatial coherence
        f0 = self.np_random.rand(self.conf['n-yaxis'], self.conf['n-xaxis'])
        f0 = gaussian_filter(f0, sigma=1.0)
        f0 = np.clip(f0, 0.0, 1.0)
               
        tmp[..., 0] = f0
        i_slow = 0
        for t in range(1, self.conf['max_episode_iteration']):
            prev_f = tmp[..., t-1]

            alpha = self.conf['alpha_ground_truth']
            if i_slow < self.conf['alpha_slow_freq']:
                alpha = self.conf['alpha_slow_ground_truth']
                i_slow += 1
            else:
                i_slow = 0

            tmp[..., t] = self.f_truth(prev_f, alpha)


        self.ground_truth = tmp
            
            
