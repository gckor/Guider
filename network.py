import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from distributions import TanhNormal
import pytorch_util as ptu
from env_util import map_goal_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentSubgoal(nn.Module):
    def __init__(
        self,
        goal_dim,
        latent_dim,
        enc_hidden_size=256,
        dec_hidden_size=256,
        init_w=1e-3,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
    ):
        super(LatentSubgoal, self).__init__()

        # Encoder
        self.encs = []
        self.enc_hidden_size = [enc_hidden_size] * 3
        self.dec_hidden_size = [dec_hidden_size] * 3
        in_size = 2 * goal_dim

        for i, next_size in enumerate(self.enc_hidden_size):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__('enc{}'.format(i), fc)
            self.encs.append(fc)

        self.enc_mean = nn.Linear(self.enc_hidden_size[-1], latent_dim)
        self.enc_mean.weight.data.uniform_(-init_w, init_w)
        self.enc_mean.bias.data.uniform_(-init_w, init_w)
        self.enc_log_std = nn.Linear(self.enc_hidden_size[-1], latent_dim)
        self.enc_log_std.weight.data.uniform_(-init_w, init_w)
        self.enc_log_std.bias.data.uniform_(-init_w, init_w)

        # Decoder
        self.decs = []
        self.dec_dropouts = []
        
        in_size = goal_dim + latent_dim

        for i, next_size in enumerate(self.dec_hidden_size):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__('dec{}'.format(i), fc)
            self.decs.append(fc)

            dropout = nn.Dropout(0.1)
            self.__setattr__('dec_dropout{}'.format(i), dropout)
            self.dec_dropouts.append(dropout)
        
        self.dec_out = nn.Linear(self.dec_hidden_size[-1], goal_dim)
        self.dec_out.weight.data.uniform_(-init_w, init_w)
        self.dec_out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, c_next_state):        
        h = torch.cat((state, c_next_state), dim=-1)

        for enc in self.encs:
            h = F.gelu(enc(h))
        
        z_mean = self.enc_mean(h)
        z_log_std = self.enc_log_std(h).clamp(-5, 2)
        z_std = torch.exp(z_log_std)
        z = z_mean + z_std * torch.randn_like(z_std)
        return z, z_mean, z_std

    def decode(self, state, z):
        h = torch.cat((state, z), dim=-1)
        
        for i, dec in enumerate(self.decs):
            h = F.gelu(dec(h))
            h = self.dec_dropouts[i](h)
        
        subgoal = self.dec_out(h)
        return subgoal


class Prior(nn.Module):
    def __init__(
        self,
        goal_dim,
        latent_dim,
        hidden_size=256,
        init_w=1e-3,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1
    ):
        super(Prior, self).__init__()

        self.hidden_size = [hidden_size] * 3
        self.fcs = []
        in_size = goal_dim

        for i, next_size in enumerate(self.hidden_size):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
        
        self.mean = nn.Linear(in_size, latent_dim)
        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)
        self.log_std = nn.Linear(in_size, latent_dim)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        h = state

        for fc in self.fcs:
            h = F.relu(fc(h))
        
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        return mean, std


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        level='high',
        use_tanh=False,
        hidden_size=256,
        init_w=1e-3,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        **kwargs,
    ):
        super(Actor, self).__init__()

        self.level = level
        self.use_tanh = use_tanh
        self.hidden_size = [hidden_size] * 3
        self.fcs = []

        if level == 'high':
            in_size = 2 * goal_dim
        if level == 'low':
            in_size = state_dim + 2 * goal_dim

        for i, next_size in enumerate(self.hidden_size):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
        
        self.mean = nn.Linear(in_size, action_dim)
        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)
        self.log_std = nn.Linear(in_size, action_dim)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, goal, subgoal=None):
        if self.level == 'high':
            state = torch.cat((state, goal), dim=-1)
        if self.level == 'low':
            state = torch.cat((state, subgoal, goal), dim=-1)

        h = state

        for fc in self.fcs:
            h = F.relu(fc(h))
        
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = torch.exp(log_std)

        if self.use_tanh:
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh_value = tanh_normal.rsample(
                return_pretanh_value=True
            )
            log_prob = tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            normal = Normal(mean, std)
            action = mean + std * torch.randn_like(std)
            log_prob = normal.log_prob(action).sum(dim=1, keepdim=True)
 
        return action, mean, std, log_prob


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        level='high',
        hidden_size=256,
        init_w=1e-3,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        **kwargs,
    ):
        super(Critic, self).__init__()
        self.level = level
        self.hidden_size = [hidden_size] * 3
        self.fcs = []

        in_size = state_dim + action_dim

        if level == 'high':
            in_size = action_dim + 2 * goal_dim
        if level == 'low':
            in_size = state_dim + action_dim + 2 * goal_dim

        for i, next_size in enumerate(self.hidden_size):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
        
        self.last_fc = nn.Linear(in_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, goal, subgoal=None):
        if self.level == 'high':
            state = torch.cat((state, goal), dim=-1)        
        if self.level == 'low':
            state = torch.cat((state, subgoal, goal), dim=-1)
            
        h = torch.cat((state, action), dim=-1)

        for fc in self.fcs:
            h = F.relu(fc(h))

        q = self.last_fc(h)
        return q
