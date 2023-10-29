from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1 = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.act  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew  = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, next_state, reward, done):
        assert state.size(0) == action.size(0) == next_state.size(0) == reward.size(0) == done.size(0)
        ep_len = state.size(0)
        if (self.ptr+ep_len) > self.max_size:
            exceed = self.max_size - (self.ptr + ep_len)    # negative value
            self.obs1[self.ptr:] = state[:exceed]
            self.obs2[self.ptr:] = next_state[:exceed]
            self.act[self.ptr:]  = action[:exceed]
            self.rew[self.ptr:]  = reward[:exceed]
            self.done[self.ptr:] = done[:exceed]

            self.obs1[:-exceed] = state[exceed:]
            self.obs2[:-exceed] = next_state[exceed:]
            self.act[:-exceed]  = action[exceed:]
            self.rew[:-exceed]  = reward[exceed:]
            self.done[:-exceed] = done[exceed:]
        else:
            self.obs1[self.ptr:self.ptr+ep_len] = state
            self.obs2[self.ptr:self.ptr+ep_len] = next_state
            self.act[self.ptr:self.ptr+ep_len]  = action
            self.rew[self.ptr:self.ptr+ep_len]  = reward
            self.done[self.ptr:self.ptr+ep_len] = done

        self.ptr = (self.ptr+ep_len) % self.max_size
        self.size = min(self.size+ep_len, self.max_size)
        return ep_len

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs1 = torch.as_tensor(self.obs1[idxs]),
            obs2 = torch.as_tensor(self.obs2[idxs]),
            act  = torch.as_tensor(self.act[idxs]),
            rew  = torch.as_tensor(self.rew[idxs]),
            done = torch.as_tensor(self.done[idxs])
        )
        return batch

class TD3:
    def __init__(self, actor_critic, gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, 
                 action_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.ac = actor_critic
        self.ac_targ = deepcopy(self.ac)
        self.gamma = gamma
        self.polyak = polyak
        self.policy_delay = policy_delay
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.n_update = 0

        # Freeze target network
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        
        # optimizer
        self.pi_optim = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optim = Adam(self.q_params, lr=q_lr)

    def compute_loss_q(self, batch):
        o1, a, r, o2, d = batch['obs1'], batch['act'], batch['rew'], batch['obs2'], batch['done']

        q1 = self.ac.q1(o1, a)
        q2 = self.ac.q2(o1, a)

        # Compute Bellman backup
        with torch.no_grad():
            # Generate pi_target with noise
            pi_targ = self.ac_targ.pi(o2)
            noise = (torch.randn_like(pi_targ) * self.target_noise).clamp_(-self.noise_clip, self.noise_clip)
            a2 = (pi_targ + noise).clamp_(-1.0, 1.0)    # limit action to [-1.0, 1.0]

            # target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma*(1-d)*(q_pi_targ)
        # MSE loss
        loss_q1 = ((q1-backup)**2).mean()
        loss_q2 = ((q2-backup)**2).mean()
        loss_q  = loss_q1 + loss_q2
        q_info = dict(LossQ=loss_q.detach().numpy(), Q1=q1.detach().numpy(), Q2=q2.detach().numpy())
        return loss_q, q_info

    def compute_loss_pi(self, batch):
        o1 = batch['obs1']
        a1 = self.ac.pi(o1)
        q1_pi = self.ac.q1(o1, a1)
        return -q1_pi.mean()

    def update(self, batch):
        self.n_update += 1
        # Update Q1 and Q2
        self.q_optim.zero_grad()
        loss_q, q_info = self.compute_loss_q(batch=batch)
        loss_q.backward()
        self.q_optim.step()

        if self.n_update % self.policy_delay == 0:
            # Freeze Q-network
            for p in self.q_params:
                p.require_grad = False

            # update pi
            self.pi_optim.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(batch=batch)
            loss_pi.backward()
            self.pi_optim.step()

            # Unfreeze Q-network
            for p in self.q_params:
                p.require_grad = True

            # Update target network with polyak averaging
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1.-self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        with torch.no_grad():
            a = self.ac.pi( torch.as_tensor(o, dtype=torch.float32) ).numpy()
            a += noise_scale * torch.randn_like(a)
            return a.clamp_(a, -1.0, 1.0).numpy()
