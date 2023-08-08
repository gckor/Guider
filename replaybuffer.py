import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    def __init__(self, env_name, epi_len, state_dim, action_dim, goal_dim, max_size=50000, **kwargs):
        self.env_name = env_name
        self.max_size = max_size
        self.epi_len = epi_len
        self.goal_dim = goal_dim
        self.storage = dict()
        self.storage['state'] = np.zeros((max_size, epi_len, state_dim))
        self.storage['action'] = np.zeros((max_size, epi_len - 1, action_dim))
        self.storage['goal'] = np.zeros((max_size, epi_len - 1, goal_dim))
        self.storage['achv_goal'] = np.zeros((max_size, epi_len, goal_dim))
        self.storage['T'] = np.zeros((max_size, 1), dtype=int)
        self.size = 0
        self.ptr = 0
    
    def add(self, state, action, goal, achv_goal, T=None):
        self.storage['state'][self.ptr] = state
        self.storage['action'][self.ptr] = action
        self.storage['goal'][self.ptr] = goal
        self.storage['achv_goal'][self.ptr] = achv_goal

        if T is not None:
            self.storage['T'][self.ptr] = T

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def subgoal_sample(self, batch_size, subgoal_period, relabel_ratio=0.8):
        '''
        subgoal_period = c
        goal: relabeled with the relabel_ratio, otherwise keep original one
        return (state, nstep_next_state, goal)
        '''
        epi_idx = np.random.randint(0, self.size, size=batch_size)

        if self.env_name.startswith('kitchen'):
            epi_len = self.storage['T'][epi_idx].squeeze()
        else:
            epi_len = self.epi_len

        timestep = np.random.randint(0, epi_len - 1, size=batch_size)
        c_next_timestep = np.minimum(timestep + subgoal_period, epi_len - 1)
        future_t = np.random.randint(c_next_timestep, epi_len)
        relabel = np.random.rand(batch_size) < relabel_ratio

        state = self.storage['state'][epi_idx, timestep]
        c_next_state = self.storage['state'][epi_idx, c_next_timestep]
        goal = self.storage['goal'][epi_idx, timestep]
        goal[relabel] = self.storage['achv_goal'][epi_idx, future_t][relabel]

        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(c_next_state).to(device),
            torch.FloatTensor(goal).to(device),
        )
    
    def sample(self, batch_size, relabel_ratio, subgoal_period):
        '''
        goal: relabeled with the relabel_ratio, otherwise keep original one
        return (state, action, next_state, subgoal, goal)
        '''
        epi_idx = np.random.randint(0, self.size, size=batch_size)

        if self.env_name.startswith('kitchen'):
            epi_len = self.storage['T'][epi_idx].squeeze()
        else:
            epi_len = self.epi_len

        timestep = np.random.randint(0, epi_len - 1, size=batch_size)
        subgoal_t = np.random.randint(timestep + 1, timestep + subgoal_period + 1, size=batch_size)
        subgoal_t = np.minimum(subgoal_t, epi_len - 1)
        relabel = np.random.rand(batch_size) < relabel_ratio
        relabel_t = np.random.randint(subgoal_t, epi_len, size=batch_size)

        state = self.storage['state'][epi_idx, timestep]
        next_state = self.storage['state'][epi_idx, timestep + 1]
        action = self.storage['action'][epi_idx, timestep]
        goal = self.storage['goal'][epi_idx, timestep]
        subgoal = self.storage['achv_goal'][epi_idx, subgoal_t]
        goal[relabel] = self.storage['achv_goal'][epi_idx, relabel_t][relabel]

        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(subgoal).to(device),
            torch.FloatTensor(goal).to(device),
        )
    
    def load(self, data):
        for i in range(data['o'].shape[0]):
            if self.env_name.startswith('kitchen'):
                self.add(
                    data['o'][i],
                    data['u'][i],
                    data['g'][i],
                    data['ag'][i],
                    data['T'][i],
                )
            else:
                self.add(
                    data['o'][i],
                    data['u'][i],
                    data['g'][i],
                    data['ag'][i],
                )
        
        print("Dataset size:" + str(self.size))
