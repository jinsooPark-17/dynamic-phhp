import os
from copy import deepcopy
import itertools
import torch
from torch.optim import Adam
    
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

        q_info = dict(Q1=q1.detach().numpy(), Q2=q2.detach().numpy()) # LossQ=loss_q.detach().numpy()

        return loss_q, q_info
    
    def compute_loss_pi(self, batch):
        o = batch['obs1']
        pi, logp_pi = self.ac.pi(o, with_logprob=True)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        p_info = dict(LogPi=logp_pi.detach().numpy()) # LossPi=loss_pi.detach().numpy()

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
        
        return q_info, pi_info

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

    def get_action(self, o, noise_scale):
        with torch.no_grad():
            return self.ac.pi( torch.as_tensor(o, dtype=torch.float32), deterministic=False, noise_scale=noise_scale ).numpy()

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
        loss_pi = -q1_pi.mean()
        pi_info = dict(LossPi=loss_pi.detach().numpy())

        return loss_pi, pi_info

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
        return q_info, pi_info

    def update_mpi(self, batch, comm):
        self.n_update += 1

        # Update Q1 and Q2
        self.q_optim.zero_grad()
        loss_q, q_info = self.compute_loss_q(batch=batch)
        loss_q.backward()
        # Average gradient across MPI jobs
        for p in self.q_params:
            p_grad_numpy = p.grad.numpy()
            avg_p_grad = comm.allreduce(p.grad) / comm.Get_size()
            p_grad_numpy[:] = avg_p_grad[:]
        self.q_optim.step()

        if self.n_update % self.policy_delay == 0:
            # Freeze Q-network
            for p in self.q_params:
                p.require_grad = False

            # Update PI
            self.pi_optim.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(batch=batch)
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
        
        return q_info, pi_info

