import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1 = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.act  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew  = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs1, act, obs2, rew, done):
        self.obs1[self.ptr] = obs1
        self.obs2[self.ptr] = obs2
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.done[self.ptr] = done

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

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

class EpisodeReplayBuffer:
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