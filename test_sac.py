import os
import time
import torch
import argparse
import subprocess
from script.network import MLPActorCritic
from rl.sac import ReplayBuffer, SAC
from torch.utils.tensorboard import SummaryWriter

def run_episode(output_path, model_path, explore=False, test=False):
    commands = ["python3", "script/gym_episode.py", output_path, "--network_dir", model_path]
    if explore:
        commands += ["--explore"]
    elif test:
        commands += ["--test"]
    return subprocess.Popen(commands)

def load_data(sample_path):
    data = torch.load(sample_path)
    return (
        torch.from_numpy(data['state']).to(torch.float32), 
        torch.from_numpy(data['action']).to(torch.float32), 
        torch.from_numpy(data['next_state']).to(torch.float32), 
        torch.from_numpy(data['reward']).to(torch.float32), 
        torch.from_numpy(data['done']).to(torch.float32) 
    )

OBS_DIM = 3
ACT_DIM = 1
STORAGE       = "network"
MODEL_STORAGE = os.path.join(STORAGE, "pi.pt")
TRAIN_SAMPLE  = "/tmp/train_sample.pt"
TEST_SAMPLE   = "/tmp/test_sample.pt"

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

    tensorboard = SummaryWriter("tensorboard/test")
    replay = ReplayBuffer(obs_dim=OBS_DIM, act_dim=ACT_DIM, size=args.replay_size)
    sac = SAC(actor_critic=MLPActorCritic(n_obs=OBS_DIM, n_act=ACT_DIM, hidden=(256,256,256,)),
              gamma=args.gamma, polyak=args.polyak, lr=args.lr, alpha=args.alpha)
    sac.save(STORAGE)

    # Main loop
    ## Run first pure exploration episode & test episode
    train_episode = run_episode(output_path=TRAIN_SAMPLE, model_path=MODEL_STORAGE, explore=True)
    train_episode.wait()

    total_steps = 0
    for epoch in range(100):
        start_time = time.time()
        while total_steps // 1000 == epoch:
            train_episode = run_episode(output_path=TRAIN_SAMPLE, model_path=MODEL_STORAGE, explore=(True if total_steps < args.explore_steps else False))

            # Store episode information to replay buffer and tensorboard
            s, a, ns, r, d = load_data(TRAIN_SAMPLE)
            ep_len = replay.store(s, a, ns, r, d)
            total_steps += ep_len
            tensorboard.add_scalars('reward', {'train': r.sum()}, total_steps)

            if total_steps > args.update_after:
                for k in range(total_steps-ep_len, total_steps):
                    q_info, pi_info = sac.update(batch=replay.sample_batch(args.batch_size))
                    tensorboard.add_scalar("Loss/Pi", pi_info.detach().item(), k)
                    tensorboard.add_scalar("Loss/Q",  q_info.detach().item(), k)
                sac.save(STORAGE)
            
            # wait for the next episode
            train_episode.wait()

        # When one episode ends, run test episode
        test_episode = run_episode(output_path=TEST_SAMPLE, model_path=MODEL_STORAGE, test=True)
        test_episode.wait()
        _, _, _, r, d = load_data(TEST_SAMPLE)
        tensorboard.add_scalars("reward", {'test': r.sum()}, total_steps)
        print(f"Epoch {epoch+1} took {time.time() - start_time:.2f} seconds.")