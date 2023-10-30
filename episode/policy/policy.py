import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
LOG2 = 0.6931471805599453

class Actor(nn.Module):
    def __init__(self, n_scan, n_act=3):
        super().__init__()

        # Store info
        self.n_scan = n_scan

        # Define network structure
        self.conv_net = nn.Sequential(
            nn.Conv1d(n_scan, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32*160, 256, dtype=torch.float32),
            nn.ReLU(),
        )
        """
        # Convolution network with 1x1 conv layer
        # 2x640 -> 32x320 -> 32x160 -> 1x160
        self.conv_net = nn.Sequential(
            nn.Conv1d(n_scan, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(32,1,1),  # Add 1x1 Conv layer
            nn.Flatten(start_dim=1),
            nn.Linear(160, 256, dtype=torch.float32),
            nn.ReLU(),
        )
        """
        self.linear_net = nn.Sequential(
            nn.Linear(256+3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, n_act)
        self.log_std_layer = nn.Linear(256, n_act)

    def forward(self, x, deterministic=False, with_logprob=False):
        scan, features = torch.split(x, 640*self.n_scan, dim=-1)
        scan = scan.view(-1, self.n_scan, 640)

        # Extract features from input
        scan = self.conv_net( scan )
        x = torch.concat([scan, features], dim=-1)
        x = self.linear_net( x )

        # Sample action from normal distribution
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        pi_distribution = torch.distributions.normal.Normal(mu, std)

        if deterministic:
            action = mu
        else:
            action = pi_distribution.rsample()

        if with_logprob:
            logp = pi_distribution.log_prob(action).sum(axis=-1)
            logp -= (2*(LOG2 - action - F.softplus(-2*action))).sum(axis=1)
        else:
            logp = None
        action = torch.tanh(action)

        return action, logp
    
class QFunction(nn.Module):
    def __init__(self, n_scan, n_act=3):
        super().__init__()
        self.n_scan = n_scan

        # Define network structure
        self.conv_net = nn.Sequential(
            nn.Conv1d(self.n_scan, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32*160, 256),
            nn.ReLU()
        )
        self.linear_net = nn.Sequential(
            nn.Linear(256 + 3 + n_act, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Identity()
        )

    def forward(self, obs, act):
        scan, feature = torch.split(obs, self.n_scan*640, dim=-1)
        scan = scan.view(-1, self.n_scan, 640)

        # extract features from scan image
        scan = self.conv_net( scan )
        sa = torch.concat([scan, feature, act], dim=-1)
        q = self.linear_net( sa )

        return torch.squeeze(q, -1)
    
class ActorCritic(nn.Module):
    def __init__(self, n_scan, n_act=3):
        super().__init__()

        self.pi = Actor(n_scan, n_act)
        self.q1 = QFunction(n_scan, n_act)
        self.q2 = QFunction(n_scan, n_act)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()