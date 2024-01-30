import argparse
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
        ["python3", "script/gym_episode.py", sample_path, "--network_dir", model_path, *args]
    )

def load_data(sample_path):
    data = torch.load(sample_path)
    return (
        torch.from_numpy(data['state']).to(torch.float32), 
        torch.from_numpy(data['action']).to(torch.float32), 
        torch.from_numpy(data['next_state']).to(torch.float32), 
        torch.from_numpy(data['reward']).to(torch.float32), 
        torch.from_numpy(data['done']).to(torch.float32) 
    )

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_size", type=int, default=int(3e4))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99, help="(default: 0.99)")
    parser.add_argument("--polyak", type=float, default=0.005, help="(default: 0.995)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2, help="(default: 0.2)")
    parser.add_argument("--explore_steps", type=int, default=10000)
    parser.add_argument("--update_after", type=int, default=1000)
    args = parser.parse_args()

    replay = ReplayBuffer(obs_dim=OBS_DIM, act_dim=ACT_DIM, size=args.replay_size)
    sac = SAC(actor_critic=MLPActorCritic(n_obs=OBS_DIM, n_act=ACT_DIM, hidden=(256,256,256,)),
              gamma=args.gamma, polyak=args.polyak, lr=args.lr, alpha=args.alpha)
    sac.save(MODEL_DIR)

    # Main loop
    ## Run first episode
    ep_proc = run_episode(sample_path='/tmp/sample.npz', model_path="network/pi.pt", explore=True)
    ep_proc.wait()

    total_steps = 0
    while total_steps < 1e+5:
        ep_proc = run_episode(sample_path='/tmp/sample.npz', model_path="network/pi.pt", explore=(True if total_steps < args.explore_steps else False))

        s, a, ns, r, d = load_data("/tmp/sample.npz")
        replay.store(s, a, ns, r, d)
        ep_len, ep_rew = r.size(0), r.sum()
        total_steps += ep_len

        print(f"Episode reward: {ep_rew}")  # LOG

        if total_steps > args.update_after:
            for k in range(ep_len):
                q_info, pi_info = sac.update(batch=replay.sample_batch(args.batch_size))
                print('\t', q_info, pi_info)

            # Save new model
            sac.save(MODEL_DIR)

        # wait for the next episode
        ep_proc.wait()