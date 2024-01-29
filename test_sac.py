import torch
import subprocess
from script.network import MLPActorCritic
from rl.sac import ReplayBuffer, SAC
from torch.utils.tensorboard import SummaryWriter

OBS_DIM = 3
ACT_DIM = 1
MODEL_DIR = "network"

def run_episode(sample_path, model_path, explore=False):
    args = []
    if explore:
        args += ["--explore"]
    return subprocess.Popen(
        ["python3", "script/gym_episode.py", sample_path, network_path, *args]
    )

def load_data(smaple_path):
    data = np.load(sample_path)
    return (
        torch.from_numpy(data['state'], dtype=torch.float32), 
        torch.from_numpy(data['action'], dtype=torch.float32), 
        torch.from_numpy(data['next_state'], dtype=torch.float32), 
        torch.from_numpy(data['reward'], dtype=torch.float32), 
        torch.from_numpy(data['done', dtype=torch.float32]) 
    )

if __name__ == '__main__':
    
    replay = ReplayBuffer(obs_dim=OBS_DIM, act_dim=ACT_DIM, size=int(1e+4))
    sac = SAC(actor_critic=MLPActorCritic(n_obs=OBS_DIM, n_obs=ACT_DIM),)
            #   gamma=args.gamma, polyak=args.polyak, lr=args.lr, alpha=args.alpha)
    sac.save(MODEL_DIR)

    # Main loop
    ## Run first episode
    ep_proc = run_episode(sample_path='/tmp/sample.npz', model_path="network/pi.pt"explore=True)
    ep_proc.wait()

    total_steps = 0
    while total_steps < 1e+5:
        ep_proc = run_episode(sample_path='/tmp/sample.npz', model_path="network/pi.pt"explore=(True if total_steps < 2000 else False))

        s, a, ns, r, d = load_data("/tmp/sample.npz")
        replay.store(s, a, ns, r, d)
        ep_len, ep_rew = r.size(0), r.sum()
        total_steps += ep_len

        print(f"Episode reward: {ep_rew}")  # LOG

        if total_steps > 5000:
            for k in range(ep_len):
                q_info, pi_info = sac.update(batch=replay.sample(32))
                print('\t', q_info, pi_info)

            # Save new model
            sac.save(MODEL_DIR)

        # wait for the next episode
        ep_proc.wait()