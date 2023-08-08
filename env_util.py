import gym
from gym.core import Wrapper
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kitchen
OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }

def make_env(env_name, seed):
    env = gym.make(env_name)
    env_info = dict()

    if env_name.startswith('antmaze'):
        env_info['epi_len'] = 1001
        env_info['goal_dim'] = 2
    if env_name.startswith('Fetch'):
        env = FetchGoalWrapper(env)
        env_info['epi_len'] = 50
        env_info['goal_dim'] = 3
    if env_name.startswith('kitchen'):
        env_info['epi_len'] = 280
        env_info['goal_dim'] = 21
    
    env.seed(seed)
    env.action_space.seed(seed)

    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        env_info['state_dim'] = env.observation_space['observation'].shape[0]        
    else:
        env_info['state_dim'] = env.observation_space.shape[0]
 
    env_info['action_dim'] = env.action_space.shape[0]
    env_info['max_action'] = float(env.action_space.high[0])

    return env, env_info

def get_reward(next_state, goal, env_name, env):
    next_state = map_goal_space(env_name, next_state)
    calc_distance = nn.PairwiseDistance()

    if env_name.startswith('antmaze'):
        epsilon = 0.5
    if env_name.startswith('Fetch'):
        epsilon = 0.05
    if env_name.startswith('Hand'):
        epsilon = 0.01
    if env_name.startswith('kitchen'):
        epsilon = 0.3

    if env_name.startswith('kitchen'):
        reward = 0
        offset = 9
        for element in env.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element] - offset
            reward += calc_distance(next_state[:, element_idx], goal[:, element_idx]) < epsilon
    else:
        reward = 1. * (calc_distance(next_state, goal) < epsilon) - 1.        

    return reward.unsqueeze(dim=1)

def map_goal_space(env_name, state):
        '''
        State Shape: (Batch size * state dimension)
        Goal Shape: (Batch size * goal dimension)
        '''
        if env_name.startswith('antmaze'):
            s_to_g = state[:, :2]
        if env_name.startswith('Fetch'):
            s_to_g = state[:, 3:6]
        if env_name.startswith('FetchReach'):
            s_to_g = state[:, :3]
        if env_name.startswith('kitchen'):
            s_to_g = state[:, 9:30]

        return s_to_g

def reach_goal(current, target, env_name, env):
    if isinstance(current, np.ndarray):
        current = torch.FloatTensor(current).reshape(1, -1).to(device)

    calc_distance = nn.PairwiseDistance()

    if env_name.startswith('antmaze'):
        epsilon = 1.0
    if env_name.startswith('Fetch'):
        epsilon = 0.05
    if env_name.startswith('kitchen'):
        epsilon = 0.3
        reward = 0
        offset = 9
        for element in env.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element] - offset
            reward += calc_distance(current[:, element_idx], target[:, element_idx]) < epsilon
        return reward.item() == 4

    return calc_distance(current, target) < epsilon

def compute_kl(post_mean, post_std, prior_mean, prior_std):
    kl = torch.log(prior_std) - torch.log(post_std) + 0.5 * ((post_std.pow(2) + (post_mean - prior_mean).pow(2)) / prior_std.pow(2) - 1)
    return kl.mean()

class FetchGoalWrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def compute_rewards(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_rewards(achieved_goal, desired_goal, info)
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render()
    
    def sample_goal(self):
        import pdb;pdb.set_trace
        return self.env.env._sample_goal()
