import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2 = 0.6931471805599453
class Actor(nn.Module):
    def __init__(self, n_scan):
        super().__init__()

        # Store info
        self.n_scan = n_scan
        n_plan = 10

        # Define network structure
        self.conv1 = nn.Conv1d(n_scan, 32, 5, 2, 2, dtype=torch.float32)
        self.conv2 = nn.Conv1d(32, 32, 3, 2, 1, dtype=torch.float32)
        self.conv3 = nn.Linear(32*160, 256, dtype=torch.float32)

        self.fc1 = nn.Linear(256 + 2 + 1, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mu_layer = nn.Linear(256, 3)
        self.log_std_layer = nn.Linear(256, 3)

    def forward(self, x, deterministic=False, with_logprob=True):
        scan, features = torch.split(x, 640*self.n_scan, dim=-1)
        scan = scan.view(-1, self.n_scan, 640)

        # extract features from scans
        scan = F.relu( self.conv1(scan) )
        scan = F.relu( self.conv2(scan) )
        scan = torch.flatten(scan, 1)
        scan = F.relu( self.conv3(scan) )

        # Concatenate scan features with other info
        x = torch.concat([scan.squeeze(), features], dim=-1)
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )

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
        # action = action * 1.0

        return action, logp
    
class QFunction(nn.Module):
    def __init__(self, n_scan=1):
        super().__init__()
        self.n_scan = n_scan

        # Define network structure
        self.conv1 = nn.Conv1d(self.n_scan, 32, 5, 2, 2, dtype=torch.float32)
        self.conv2 = nn.Conv1d(32, 32, 3, 2, 1, dtype=torch.float32)
        self.conv3 = nn.Linear(32*160, 256, dtype=torch.float32)

        self.fc1 = nn.Linear(256 + 2 + 1, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, act):
        scan, feature = torch.split(obs, self.n_scan*640, dim=-1)
        scan = scan.view(-1, self.n_scan, 640)

        # extract features from scan image
        scan = F.relu( self.conv1(scan) )
        scan = F.relu( self.conv2(scan) )
        scan = torch.flatten(scan, 1)
        scan = F.relu( self.conv3(scan) )

        # concatenate scan features, feature and action
        q = torch.concat([scan, feature, act], dim=-1)
        q = F.relu( self.fc1(q) )
        q = F.relu( self.fc2(q) )
        q = F.relu( self.fc3(q) )

        return torch.squeeze(q, -1)
    
class ActorCritic(nn.Module):
    def __init__(self, n_scan):
        super().__init__()

        self.pi = Actor(n_scan)
        self.q1 = QFunction(n_scan)
        self.q2 = QFunction(n_scan)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()