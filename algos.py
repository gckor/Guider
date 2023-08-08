import numpy as np
import torch
import torch.nn.functional as F
import pytorch_util as ptu
from logger import create_stats_ordered_dict
from network import LatentSubgoal, Prior, Actor, Critic
from env_util import map_goal_space, get_reward, reach_goal, compute_kl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LatentSubgoalModel(object):
    def __init__(
        self,
        replay_buffer,
        goal_dim,
        latent_dim,
        latentmodel_lr=3e-4,
        prior_lr=3e-4,
        subgoal_period=50,
        env_name=None,
        beta=0.1,
        kl_balance=0.8,
        **kwargs,
    ):
        self.replay_buffer = replay_buffer
        self.latentmodel = LatentSubgoal(goal_dim, latent_dim).to(device)
        self.latentmodel_optimizer = torch.optim.Adam(self.latentmodel.parameters(), lr=latentmodel_lr)
        self.prior = Prior(goal_dim, latent_dim).to(device)
        self.prior_optimizer = torch.optim.Adam(self.prior.parameters(), lr=prior_lr)

        self.subgoal_period = subgoal_period
        self.beta = beta
        self.kl_balance = kl_balance
        self.env_name = env_name
        
    def train(self, save_dir, logger, pre_batch_size, pre_total_step, pre_log_period, pre_save_period, **kwargs):
        for step in range(1, pre_total_step + 1):
            # Train
            loss_info = self.train_step(pre_batch_size)

            # Log
            if step % pre_log_period == 0:
                self.log(logger, step, loss_info)

            # Save Model
            if step % pre_save_period == 0:
                self.save(step, save_dir)

    def train_step(self, batch_size=100):
        state, c_next_state, _ = self.replay_buffer.subgoal_sample(batch_size, self.subgoal_period)
        state = map_goal_space(self.env_name, state)
        c_next_state = map_goal_space(self.env_name, c_next_state)
        elbo_loss, prior_loss, loss_info = self.loss(state, c_next_state)
        self.latentmodel.train()

        # Update Encoder, Decoder
        self.latentmodel_optimizer.zero_grad()
        elbo_loss.backward()
        self.latentmodel_optimizer.step()

        # Update Prior Network
        self.prior_optimizer.zero_grad()
        prior_loss.backward()
        self.prior_optimizer.step()

        return loss_info
    
    def loss(self, state, c_next_state):
        loss_info = dict()
        z, z_mean, z_std = self.latentmodel(state, c_next_state)
        prior_mean, prior_std = self.prior(state)

        # Reconstruction Loss
        recon = self.latentmodel.decode(state, z)
        recon_loss = F.mse_loss(recon, c_next_state)
        loss_info['recon_loss'] = recon_loss.item()

        # KL Loss
        regul_loss = compute_kl(z_mean, z_std, prior_mean.detach(), prior_std.detach())
        prior_loss = compute_kl(z_mean.detach(), z_std.detach(), prior_mean, prior_std)
        loss_info['regul_loss'] = regul_loss.item()
        loss_info['prior_loss'] = prior_loss.item()

        # Calculate Loss
        elbo_loss = recon_loss + self.beta * (1 - self.kl_balance) * regul_loss
        prior_loss = self.beta * self.kl_balance * prior_loss
        loss_info['elbo_loss'] = elbo_loss.item()

        return elbo_loss, prior_loss, loss_info

    def log(self, logger, step, loss_info):
        logger.record_tabular('Training steps', step)

        for key, val in loss_info.items():
            logger.record_tabular(key, val)        

        logger.dump_tabular()

    def save(self, filename, directory):
        torch.save(self.latentmodel.state_dict(), '%s/latentmodel_%s.pth' % (directory, filename))
        torch.save(self.prior.state_dict(), '%s/subgoalprior_%s.pth' % (directory, filename))

    def load(self, latentmodel_path, prior_path):
        self.latentmodel.load_state_dict(torch.load(latentmodel_path, map_location=device))
        self.prior.load_state_dict(torch.load(prior_path, map_location=device))


class HighlevelModel(object):
    def __init__(
        self,
        env,
        replay_buffer,
        latentmodel,
        prior,
        state_dim,
        latent_dim,
        goal_dim,
        epi_len,
        env_name,

        subgoal_period=50,
        relabel_ratio=0.8,
        
        high_actor_lr=1e-4,
        high_critic_lr=1e-4,        
        
        # Automatic Coefficient Learning
        use_automatic_alpha_tuning=True,
        target_divergence=1,
        alpha=1,
        with_lagrange=True,
        lagrange_thresh=5,
        alpha_prime=1,

        # CQL
        reward_scale=1.0,
        tau=0.005,
        discount=0.99,

        ## sort of backup
        max_q_backup=True,
        num_random=10,

        **kwargs,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.latentmodel = latentmodel
        self.latentmodel.eval()
        self.subgoalprior = prior
        
        self.actor = Actor(state_dim, latent_dim, goal_dim, level='high').to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=high_actor_lr)

        self.qf1 = Critic(state_dim, latent_dim, goal_dim, level='high').to(device)
        self.qf2 = Critic(state_dim, latent_dim, goal_dim, level='high').to(device)
        self.target_qf1 = Critic(state_dim, latent_dim, goal_dim, level='high').to(device)
        self.target_qf2 = Critic(state_dim, latent_dim, goal_dim, level='high').to(device)
        self.qf_optimizer = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=high_critic_lr
        )

        self.use_automatic_alpha_tuning = use_automatic_alpha_tuning
        if self.use_automatic_alpha_tuning:
            self.target_divergence = target_divergence 
            self.log_alpha = ptu.zeros(1, torch_device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=high_actor_lr
            )
        else:
            self.alpha = alpha

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, torch_device=device, requires_grad=True)
            self.alpha_prime_optimizer = torch.optim.Adam(
                [self.log_alpha_prime],
                lr=high_critic_lr,
            )
        else:
            self.alpha_prime = alpha_prime

        self.latent_dim = latent_dim
        self.epi_len = epi_len
        self.env_name = env_name

        self.subgoal_period = subgoal_period
        self.relabel_ratio = relabel_ratio

        self.reward_scale = reward_scale
        self.discount = discount
        self.tau = tau

        ## min Q
        self.max_q_backup = max_q_backup
        self.num_random = num_random

    def generate_subgoal(self, state, target_goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = map_goal_space(self.env_name, state)
        target_goal = torch.FloatTensor(target_goal.reshape(1, -1)).to(device)
        
        with torch.no_grad():         
            _, z, *_ = self.actor(state, target_goal)
            subgoal = self.latentmodel.decode(state, z)
        
        if reach_goal(subgoal, target_goal, self.env_name, self.env):
            subgoal = target_goal

        return subgoal

    def _get_tensor_values(self, obs, latents, goal=None, network=None):
        action_shape = latents.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        goal_temp = goal.unsqueeze(1).repeat(1, num_repeat, 1).view(goal.shape[0] * num_repeat, goal.shape[1])
        preds = network(obs_temp, latents, goal_temp)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, state, num_actions, goal=None, network=None):
        state_temp = state.unsqueeze(1).repeat(1, num_actions, 1).view(state.shape[0] * num_actions, state.shape[1])
        goal_temp = goal.unsqueeze(1).repeat(1, num_actions, 1).view(goal.shape[0] * num_actions, goal.shape[1])

        with torch.no_grad():
            policy_z, _, _, policy_z_log_prob = network(state_temp, goal_temp)

        return policy_z, policy_z_log_prob.view(state.shape[0], num_actions, 1)

    def train(self, save_dir, logger, high_batch_size, high_total_step, high_log_period, high_save_period, **kwargs):
        for step in range(1, high_total_step + 1):
            # Train
            loss_info, train_info = self.train_step(high_batch_size)

            # Log
            if step % high_log_period == 0:
                self.log(logger, step, loss_info, train_info)

            # Save Model
            if step % high_save_period == 0:
                self.save(step, save_dir)
        
    def train_step(self, batch_size):
        # Sample batch
        state, c_next_state, goal = self.replay_buffer.subgoal_sample(batch_size, self.subgoal_period, self.relabel_ratio)
        reward = get_reward(c_next_state, goal, self.env_name, self.env)
        state = map_goal_space(self.env_name, state)
        c_next_state = map_goal_space(self.env_name, c_next_state)
       
        with torch.no_grad():
            _, z, _ = self.latentmodel(state, c_next_state)

        # Train
        actor_loss, total_qf_loss, loss_info, train_info = self.loss(state, z, reward, c_next_state, goal)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.qf_optimizer.zero_grad()
        total_qf_loss.backward()
        self.qf_optimizer.step()

        for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return loss_info, train_info

    def loss(self, state, z, reward, c_next_state, goal):
        loss_info = dict()
        train_info = dict()

        # Actor Loss
        policy_z, policy_z_mean, policy_z_std, _ = self.actor(state, goal)
        q_policy_z = torch.min(self.qf1(state, policy_z, goal), self.qf2(state, policy_z, goal))
        
        with torch.no_grad():
            prior_mean, prior_std = self.subgoalprior(state)
        
        kl_loss = compute_kl(policy_z_mean, policy_z_std, prior_mean, prior_std)

        if self.use_automatic_alpha_tuning:
            alpha = torch.clamp(self.log_alpha.exp(), min=0.0, max=1000000.0)
            alpha_loss = (alpha * (self.target_divergence - kl_loss).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = torch.FloatTensor([self.alpha]).to(device)

        loss_info['kl_loss'] = kl_loss.item()
        loss_info['alpha'] = alpha.item()

        actor_loss = (-q_policy_z + alpha * kl_loss).mean()
        loss_info['actor_loss'] = actor_loss.item()
        train_info['policy_z'] = policy_z.detach().cpu().numpy()
        
        # Critic Loss
        q1_pred = self.qf1(state, z, goal)
        q2_pred = self.qf2(state, z, goal)

        with torch.no_grad():
            policy_next_z, *_ = self.actor(c_next_state, goal)

            if self.max_q_backup:
                next_z_temp, _ = self._get_policy_actions(c_next_state, num_actions=10, goal=goal, network=self.actor)
                target_qf1_values = self._get_tensor_values(c_next_state, next_z_temp, goal, network=self.target_qf1).max(1)[0].view(-1, 1)
                target_qf2_values = self._get_tensor_values(c_next_state, next_z_temp, goal, network=self.target_qf2).max(1)[0].view(-1, 1)
                target_q_values = torch.min(target_qf1_values, target_qf2_values)
            else:
                target_q_values = torch.min(self.target_qf1(c_next_state, policy_next_z, goal), self.target_qf2(c_next_state, policy_next_z, goal))

        q_target = self.reward_scale * reward + self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, q_target).mean()
        qf2_loss = F.mse_loss(q2_pred, q_target).mean()
 
        random_z = torch.FloatTensor(state.shape[0] * self.num_random, z.shape[-1]).uniform_(z.min(), z.max()).to(device)
        curr_z, curr_log_prob = self._get_policy_actions(state, num_actions=self.num_random, goal=goal, network=self.actor)
        next_z, next_log_prob = self._get_policy_actions(c_next_state, num_actions=self.num_random, goal=goal, network=self.actor)
        random_density = np.log(0.5 ** curr_z.shape[-1])
        q1_rand = self._get_tensor_values(state, random_z, goal, network=self.qf1)
        q2_rand = self._get_tensor_values(state, random_z, goal, network=self.qf2)
        q1_curr_z = self._get_tensor_values(state, curr_z, goal, network=self.qf1)
        q2_curr_z = self._get_tensor_values(state, curr_z, goal, network=self.qf2)
        q1_next_z = self._get_tensor_values(state, next_z, goal, network=self.qf1)
        q2_next_z = self._get_tensor_values(state, next_z, goal, network=self.qf2)

        cat_q1 = torch.cat((q1_rand - random_density, q1_next_z - next_log_prob, q1_curr_z - curr_log_prob), dim=1)
        cat_q2 = torch.cat((q2_rand - random_density, q2_next_z - next_log_prob, q2_curr_z - curr_log_prob), dim=1)

        min_qf1_loss = torch.logsumexp(cat_q1, dim=1).mean() - q1_pred.mean()
        min_qf2_loss = torch.logsumexp(cat_q2, dim=1).mean() - q2_pred.mean()

        if self.with_lagrange:
            alpha_prime = self.log_alpha_prime.exp()
            alpha_prime_loss = -alpha_prime * ((min_qf1_loss - self.target_action_gap) + (min_qf2_loss - self.target_action_gap)).detach() * 0.5
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss.backward()
            self.alpha_prime_optimizer.step()
            alpha_prime = self.log_alpha_prime.detach().exp()
        else:
            alpha_prime = torch.FloatTensor([self.alpha_prime]).to(device)
        
        total_qf1_loss = qf1_loss + alpha_prime * min_qf1_loss
        total_qf2_loss = qf2_loss + alpha_prime * min_qf2_loss
        total_qf_loss = total_qf1_loss + total_qf2_loss

        loss_info['min_qf1_loss'] = min_qf1_loss.item()
        loss_info['min_qf2_loss'] = min_qf2_loss.item()
        loss_info['alpha_prime'] = alpha_prime.item()

        loss_info['total_qf_loss'] = total_qf_loss.item()
        train_info['q1_pred'] = q1_pred.detach().cpu().numpy()
        train_info['q2_pred'] = q2_pred.detach().cpu().numpy()
        train_info['q_target'] = q_target.detach().cpu().numpy()

        return actor_loss, total_qf_loss, loss_info, train_info
     
    def log(self, logger, step, loss_info, train_info):
        logger.record_tabular('Training steps', step)

        for key, val in loss_info.items():
            logger.record_tabular(key, val)

        for key, val in train_info.items():
            logger.record_dict(create_stats_ordered_dict(key, val))

        logger.dump_tabular()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_subgoalpolicy.pth' % (directory, filename))

    def load(self, subgoalpolicy_path):
        self.actor.load_state_dict(torch.load(subgoalpolicy_path))


class LowlevelModel(object):
    def __init__(
        self,
        env,
        replay_buffer,
        highmodel,
        state_dim,
        action_dim,
        goal_dim,
        epi_len,
        env_name,        
        low_actor_lr=0.0005,
        low_critic_lr=0.0005,
        relabel_ratio=0.8,
        subgoal_period=5,
        tau=0.005,
        discount=0.98,
        low_relabel_strategy=True,
        offset=5,
        use_tanh=False,
        **kwargs,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.highmodel = highmodel
        self.policy = Actor(state_dim, action_dim, goal_dim, level='low', use_tanh=use_tanh).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=low_actor_lr)
        self.qf1 = Critic(state_dim, action_dim, goal_dim, level='low').to(device)
        self.qf2 = Critic(state_dim, action_dim, goal_dim, level='low').to(device)
        self.target_qf1 = Critic(state_dim, action_dim, goal_dim, level='low').to(device)
        self.target_qf2 = Critic(state_dim, action_dim, goal_dim, level='low').to(device)
        self.qf_optimizer = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=low_critic_lr
        )

        self.env_name = env_name
        self.relabel_ratio = relabel_ratio
        self.subgoal_period = subgoal_period
        self.relabel_period = self.subgoal_period + offset if low_relabel_strategy else epi_len
        self.tau = tau
        self.discount = discount
    
    def train(self, save_dir, logger, low_batch_size, low_total_step, low_log_period, low_save_period, **kwargs):
        for step in range(1, low_total_step + 1):
            # Train
            loss_info = self.train_step(low_batch_size)

            # Log
            if step % low_log_period == 0:
                self.log(logger, step, loss_info)

            # Save Model
            if step % low_save_period == 0:
                self.save(step, save_dir)

    def train_step(self, batch_size):
        # Sample batch
        state, action, next_state, subgoal, goal = self.replay_buffer.sample(batch_size, self.relabel_ratio, self.relabel_period)
        reward = get_reward(next_state, subgoal, self.env_name, self.env)

        critic_loss, loss_info = self.critic_loss(state, action, next_state, reward, subgoal, goal)

        self.qf_optimizer.zero_grad()
        critic_loss.backward()
        self.qf_optimizer.step()

        for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        policy_loss, loss_info = self.policy_loss(state, action, next_state, reward, subgoal, goal)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return loss_info

    def critic_loss(self, state, action, next_state, reward, subgoal, goal):
        loss_info = dict()
        
        q1_pred = self.qf1(state, action, goal, subgoal)
        q2_pred = self.qf2(state, action, goal, subgoal)

        with torch.no_grad():
            policy_next_a, *_ = self.policy(next_state, goal, subgoal)
            target_q_values = torch.min(self.target_qf1(next_state, policy_next_a, goal, subgoal), self.target_qf2(next_state, policy_next_a, goal, subgoal))

        q_target = reward + self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)
        total_qf_loss = qf1_loss + qf2_loss
        loss_info['total_qf_loss'] = total_qf_loss.item()

        return total_qf_loss, loss_info

    def policy_loss(self, state, action, next_state, reward, subgoal, goal):
        loss_info = dict()

        policy_a, *_ = self.policy(state, goal, subgoal)

        with torch.no_grad():    
            policy_next_a, *_ = self.policy(next_state, goal, subgoal)
            q = reward + self.discount * self.qf1(next_state, policy_next_a, goal, subgoal)
            v = self.qf1(state, policy_a.detach(), goal, subgoal)
        
        adv = q - v
        weight = torch.clip(torch.exp(adv), 0, 10)

        # Policy Training
        policy_loss = (weight * F.mse_loss(policy_a, action, reduction='none')).mean()
        loss_info['policy_loss'] = policy_loss.item()
        return policy_loss, loss_info

    def log(self, logger, step, loss_info):
        logger.record_tabular('Training steps', step)

        for key, val in loss_info.items():
            logger.record_tabular(key, val)
        
        avg_reward = self.eval_policy()
        logger.record_tabular('AverageReturn', avg_reward)

        logger.dump_tabular()

    def save(self, filename, directory):
        torch.save(self.policy.state_dict(), '%s/%s_lowpolicy.pth' % (directory, filename))

    def load(self, lowpolicy_path):
        self.policy.load_state_dict(torch.load(lowpolicy_path))

    def select_action(self, state, subgoal, goal, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if isinstance(goal, np.ndarray):
            goal = torch.FloatTensor(goal).unsqueeze(0).to(device)

        with torch.no_grad():
            action, action_mean, *_ = self.policy(state, goal, subgoal)
     
            if deterministic:
                action = action_mean
                
        return action.cpu().data.numpy().flatten()
    
    def eval_policy(self, eval_episodes=100):
        avg_reward = 0.

        for _ in range(eval_episodes):
            state, done = self.env.reset(), False

            if isinstance(state, dict):
                state, target_goal = state['observation'], state['desired_goal']
            elif self.env_name.startswith('antmaze'):
                target_goal = np.array(self.env.target_goal)
            elif self.env_name.startswith('kitchen'):
                target_goal = np.array(self.env.goal)[9:30]

            while not done:
                subgoal = self.highmodel.generate_subgoal(state, target_goal)
 
                for _ in range(self.subgoal_period):
                    #Select Action
                    action = self.select_action(state, subgoal, target_goal, deterministic=True)                    
                    state, reward, done, _ = self.env.step(action)

                    if isinstance(state, dict):
                        state = state['observation']
                        reward += 1
                    avg_reward += reward

                    if done:
                        break
                    if reach_goal(map_goal_space(self.env_name, state.reshape(1, -1)), subgoal, self.env_name, self.env):
                        break

        avg_reward /= eval_episodes

        print ("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print ("---------------------------------------")

        return avg_reward
