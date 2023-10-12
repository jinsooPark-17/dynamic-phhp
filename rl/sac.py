import os
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
    
class SAC:
    def __init__(self, actor_critic, gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2):
        self.ac = actor_critic
        self.ac_targ = deepcopy(self.ac)

        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha

        # Freeze target network
        for p in self.ac_targ.parameters():
            p.require_grad = False

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Prepare optimizer
        self.pi_optim = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optim = Adam(self.q_params, lr=lr)

    def compute_loss_q(self, batch):
        o, a, r, o2, d = batch['obs1'], batch['act'], batch['rew'], batch['obs2'], batch['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(o2, with_logprob=True)

            # Target Q-value
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma*(1-d)*(q_pi_targ - self.alpha*logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1=q1.detach().numpy(), Q2=q2.detach().numpy())

        return loss_q, q_info
    
    def compute_loss_pi(self, batch):
        o = batch['obs1']
        pi, logp_pi = self.ac.pi(o, with_logprob=True)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha*logp_pi - q_pi).mean()

        p_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, p_info
    
    def update(self, batch):
        # Update Q1 and Q2
        self.q_optim.zero_grad()
        loss_q, q_info = self.compute_loss_q( batch )
        loss_q.backward()
        self.q_optim.step()

        # Freeze Q-network
        for p in self.q_params:
            p.require_grad = False

        # Update PI
        self.pi_optim.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optim.step()

        # Unfreeze Q-network
        for p in self.q_params:
            p.require_grad = True

        # Update target network by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1.-self.polyak)*p.data)
        
        return loss_q, loss_pi

    def update_mpi(self, batch, comm):
        # Update Q1 and Q2
        self.q_optim.zero_grad()
        loss_q, q_info = self.compute_loss_q( batch )
        loss_q.backward()

        # Average gradient across MPI jobs
        for p in self.q_params:
            p_grad_numpy = p.grad.numpy()
            avg_p_grad = comm.allreduce(p.grad) / comm.Get_size()
            p_grad_numpy[:] = avg_p_grad[:]
        self.q_optim.step()

        # Freeze Q-network
        for p in self.q_params:
            p.require_grad = False

        # Update PI
        self.pi_optim.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(batch)
        loss_pi.backward()

        # Average gradient across MPI jobs
        for p in self.ac.pi.parameters():
            p_grad_numpy = p.grad.numpy()
            avg_p_grad = comm.allreduce(p.grad) / comm.Get_size()
            p_grad_numpy[:] = avg_p_grad[:]
        self.pi_optim.step()

        # Unfreeze Q-network
        for p in self.q_params:
            p.require_grad = True

        # Update target network by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1.-self.polyak)*p.data)
        
        return loss_q, loss_pi

    def save(self, network_dir):
        torch.save(self.ac.pi.state_dict(), os.path.join(network_dir, "pi"))
        torch.save(self.ac.q1.state_dict(), os.path.join(network_dir, "q1"))
        torch.save(self.ac.q2.state_dict(), os.path.join(network_dir, "q2"))
        torch.save(self.pi_optim.state_dict(), os.path.join(network_dir, "pi_optimizer"))
        torch.save(self.q_optim.state_dict(),  os.path.join(network_dir, "q_optimizer"))

    def checkpoint(self, epoch, checkpoint_dir):
        torch.save(
            {
                "epoch": epoch,
                "pi": self.ac.pi.state_dict(),
                "q1": self.ac.q1.state_dict(),
                "q2": self.ac.q2.state_dict(),
                "pi_optim": self.pi_optim.state_dict(),
                "q_optim": self.q_optim.state_dict()
            }, checkpoint_dir
        )