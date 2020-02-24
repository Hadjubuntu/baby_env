import gym
from gym import spaces

import logging
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

import numpy as np

LOGGER = logging.getLogger(__name__)

default_conf = {
    # Seed for data generation
    'seed': 0,
    # Number of frames
    'n_frame': 3, # 5
    # n-xaxis (mock longitude)
    'n-xaxis': 7,
    # n-yaxis (mock latitude)
    'n-yaxis': 7,
    # max episode iteration
    'max_episode_iteration': 1000,
    # factor of ground truth modification
    'alpha_ground_truth': 0.8,
    # Minimum value of ground truth to be validated
    'validation_threshold': 0.5,
    # Sigma of gaussian filter for prediction depending on delta time
    'sigma_prediction': 1.0 # prev=0.1
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
            rew = 1.0
            validation_frame[y, x] = 0.0
            self.validation = np.expand_dims(validation_frame, -1)

        # Error pred/obs
        # print(f"Diff={self.current_obs[..., 0][y, x] - current_truth[y, x]}")

        # Update t
        self.t += 1
        
        # Update obs        
        obs = self.predict(self.t)
        obs = np.append(obs, self.validation, axis=2)
        self.current_obs = obs

        self.cum_rew += rew        
        
        # Status done
        if (self.validation == 0.0).all() or self.t >= (self.conf['max_episode_iteration'] - self.conf['n_frame']):
            done = True            
            info = {'episode': {'r': self.cum_rew, 'l': self.t}}
        
        return obs, rew, done, info
    
    def f_predict(self, truth_frame, dt):        
        pred = np.copy(truth_frame)        
        pred = gaussian_filter(pred, sigma=self.conf['sigma_prediction']*(dt+1))
        pred = np.clip(pred, 0.0, 1.0)
        
        return pred
    
    def predict(self, t):
        n_frame = self.conf['n_frame']
        pred_t_tn = np.copy(self.ground_truth[..., t:(t+n_frame)])
                
        for i in range(self.conf['n_frame']):
            pred_t_tn[..., i] = self.f_predict(pred_t_tn[..., i], i)
            
        return pred_t_tn
        
    
    def f_truth(self, prev_frame):
        """
        Apply function to previous frame to compute next frame
        """
        next_frame = np.copy(prev_frame)
        
        rand_frame = (self.np_random.rand(self.conf['n-yaxis'], self.conf['n-xaxis'])-0.5)
        next_frame = next_frame + self.conf['alpha_ground_truth'] * gaussian_filter(rand_frame, sigma=1.0)
        next_frame = np.clip(next_frame, 0.0, 1.0)
        
        return next_frame
    
    def reset_ground_truth(self):
        # Generate the first frame randomly, apply gaussian filter to imitate spatial coherence
        f0 = self.np_random.rand(self.conf['n-yaxis'], self.conf['n-xaxis'])
        f0 = gaussian_filter(f0, sigma=1.0)
               
        self.ground_truth[..., 0] = f0
        for t in range(1, self.conf['max_episode_iteration']):
            prev_f = self.ground_truth[..., t-1]
            self.ground_truth[..., t] = self.f_truth(prev_f)
            
