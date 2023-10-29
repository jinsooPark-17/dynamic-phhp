import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)

class Actor(nn.Module):
    def __init__(self, n_scan, n_act=3):
        super().__init__()
        self.n_scan = n_scan

        # Define network structure
        self.cnn = nn.Sequential(
            nn.Conv1d(n_scan, 32, 5, 2, 2), 
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2, 1), 
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32*160, 256), 
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 + 3, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, n_act), 
            nn.Tanh(),
        )

    def forward(self, obs, deterministic=False, noise_scale=0.3):
        scan, features = torch.split(obs, 640*self.n_scan, dim=-1)
        scan = scan.view(-1, self.n_scan, 640)

        # extract features from scans
        x = self.cnn(scan)
        x = torch.concat([x, features], dim=-1)
        x = self.fc(x)
        if not deterministic:
            x = (x + noise_scale * torch.randn_like(x)).clamp_(-1.0, 1.0)
        return x
    
class QFunction(nn.Module):
    def __init__(self, n_scan, n_act=3):
        super().__init__()
        self.n_scan = n_scan

        # Define network structure
        self.cnn = nn.Sequential(
            nn.Conv1d(n_scan, 32, 5, 2, 2), 
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2, 1), 
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(32*160, 256), 
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 + 3 + n_act, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 1), 
            nn.Identity(),
        )

    def forward(self, obs, act):
        scan, feature = torch.split(obs, self.n_scan*640, dim=-1)
        scan = scan.view(-1, self.n_scan, 640)

        x = self.cnn(scan)
        sa = torch.concat([x, feature, act], dim=-1)
        q = self.fc(sa)

        return torch.squeeze(q, -1)
    
class ActorCritic(nn.Module):
    def __init__(self, n_scan, n_act=3):
        super().__init__()

        self.pi = Actor(n_scan, n_act)
        self.q1 = QFunction(n_scan, n_act)
        self.q2 = QFunction(n_scan, n_act)

    def act(self, obs, deterministic=False, noise_scale=0.3):
        with torch.no_grad():
            a = self.pi(obs, deterministic, noise_scale=noise_scale)
            return a.numpy()