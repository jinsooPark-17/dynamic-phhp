import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
torch.set_default_dtype(torch.float32)
LOG2 = 0.6931471805599453

class Actor(nn.Module):
    def __init__(self, n_scan, n_plan, action_dim=3, combine=False):
        """
        input (state):
            scan: (n_scan*2, 640) np.vstack((raw_scans, hal_scans)), 
            plan: (detection_range / sensor_horizon)*2 
            vw: 2
        )
        """
        super().__init__()

        self.combine = combine
        self.n_scan = n_scan
        self.act_dim = action_dim

        if combine is True:
            self.conv_net = nn.Sequential(
                nn.Conv1d(in_channels=n_scan*2, out_channels=32, kernel_size=5, stride=2, padding=2),   # (32, 320)
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),         # (32, 160)
                nn.ReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(5120, 256),
                nn.ReLU()
            )
        else:
            # Half size
            self.conv_net = nn.Sequential(
                nn.Conv1d(in_channels=n_scan, out_channels=16, kernel_size=5, stride=2, padding=2),     # (32, 320)
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),         # (32, 160)
                nn.ReLU(),
                nn.Linear(5120, 128),
                nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(256+n_plan+2, 128),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, obs, deterministic=False, with_logprob=False):
        if isinstance(obs, dict):
            # Used during deployment
            scan = torch.from_numpy(obs['scan']).to(torch.float32).view(1,-1,640)
            plan = torch.from_numpy(obs['plan']).to(torch.float32).view(1,-1)
            vw   = torch.from_numpy(obs['vw']).to(torch.float32).view(1,-1)
        else:
            # Used during training
            if obs.dim()==1:
                obs = obs.view(1, -1)
            scan = obs[..., :self.n_scan*2*640].view(-1, self.n_scan*2, 640)
            plan = obs[..., self.n_scan*2*640:-2]
            vw   = obs[..., -2:]

        # extract features from scan
        if self.combine is True:
            scan = self.conv_net(scan)
            x = torch.concat([scan, plan, vw], dim=-1)
        else:
            scan_raw = self.conv_net(scan[:self.n_scan,:])
            scan_hal = self.conv_net(scan[self.n_scan:,:])
            x = torch.concat([scan_raw, scan_hal, plan, vw], dim=-1)
        x = self.fc(x)

        # sample action from normal distribution
        mu      = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20.0, 2.0)
        std = torch.exp(log_std)
        pi_distribution = torch.distributions.normal.Normal(mu, std)

        action = (mu if deterministic else pi_distribution.rsample())
        if with_logprob:
            logp = pi_distribution.log_prob(action).sum(axis=-1)
            logp -= (2*(LOG2 - action - F.softplus(-2*action))).sum(axis=1)
        else:
            logp = None
        
        action = torch.tanh(action).squeeze()
        return action, logp
    
class QFunction(nn.Module):
    def __init__(self, n_scan, n_plan, action_dim=3, combine=False):
        super().__init__()

        self.combine = combine
        self.n_scan = n_scan
        self.act_dim = action_dim

        if combine is True:
            self.conv_net = nn.Sequential(
                nn.Conv1d(in_channels=n_scan*2, out_channels=32, kernel_size=5, stride=2, padding=2),   # (32, 320)
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),         # (32, 160)
                nn.ReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(5120, 256),
                nn.ReLU()
            )
        else:
            # Half size
            self.conv_net = nn.Sequential(
                nn.Conv1d(in_channels=n_scan, out_channels=16, kernel_size=5, stride=2, padding=2),     # (32, 320)
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),         # (32, 160)
                nn.ReLU(),
                nn.Linear(5120, 128),
                nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(256+n_plan+2+action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Identity()
        )

    def forward(self, obs, act):
        scan = obs[:self.n_scan*2].view(self.n_scan*2, 640)
        plan = obs[self.n_scan*2:-2]
        vw   = obs[-2:]
        
        # extract features from scan
        if self.combine is True:
            scan = self.conv_net(scan)
            x = torch.concat([scan, plan, vw, act], dim=-1)
        else:
            scan_raw = self.conv_net(scan[:self.n_scan*640])
            scan_hal = self.conv_net(scan[self.n_scan*640:])
            x = torch.concat([scan_raw, scan_hal, plan, vw, act], dim=-1)

        q = self.fc(x)
        return torch.squeeze(q, -1)
    
class ActorCritic(nn.Module):
    def __init__(self, n_scan, n_plan, action_dim, combine=True):
        super().__init__()

        self.pi = Actor(n_scan, n_plan, action_dim, combine)
        self.q1 = QFunction(n_scan, n_plan, action_dim, combine)
        self.q2 = QFunction(n_scan, n_plan, action_dim, combine)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
        
if __name__ == '__main__':
    import time
    import numpy as np
    actor_critic = ActorCritic(1, 32, 3, True)

    n_state = 2*640+32+2
    n_act = 3
    s = time.time()
    x = torch.rand(n_state)
    a = actor_critic.act(x)
    print(a)

    x = torch.rand(32, n_state)
    a = actor_critic.act(x)
    print(a)

    state = dict(
            scan=np.random.rand(2, 640), 
            plan=np.random.rand(32), 
            vw=np.random.rand(2)
        )
    print( actor_critic.act(state) )
    print(time.time()-s)