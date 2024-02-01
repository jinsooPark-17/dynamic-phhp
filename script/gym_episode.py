import os
import time
import torch
import numpy as np
import argparse
import gymnasium as gym
from network import MLPActor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file_path", type=str, default="test")
    parser.add_argument("--network_dir", type=str, required=True)
    parser.add_argument("--explore", type=str)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--gui", action='store_true')
    args = parser.parse_args()

    if args.gui:
        env = gym.make("Pendulum-v1", render_mode='human')
    else:
        env = gym.make("Pendulum-v1")
    policy = MLPActor(n_obs=3, n_act=1, hidden=(256,256,256,))

    # Define data storage
    state = np.zeros((200,3))
    action = np.zeros((200,1))
    next_state = np.zeros_like(state)
    reward = np.zeros((200,))
    done = np.zeros((200,))

    try:
        while True:
            # Wait for the training process
            while os.path.exists(args.output_file_path):
                time.sleep(0.1)

            # Load new model
            policy.load_state_dict(torch.load(args.network_dir))

            # Run episode
            (s, _), d, idx = env.reset(), False, 0
            while not d:
                if not args.test and os.path.exists(args.explore):
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.from_numpy(s).to(torch.float32)
                        a, _ = policy(s, deterministic=args.test)
                        a = a.numpy()
                ns, r, terminated, truncated, _ = env.step(2.*a)
                d = (terminated or truncated)

                # Store SAS'RD
                state[idx]      = s
                action[idx]     = a
                next_state[idx] = ns
                reward[idx]     = r
                done[idx]       = d

                # Update
                s = ns
                idx += 1

            # Store episode info as a file
            episode_info = dict(state=torch.from_numpy(state[:idx]).to(torch.float32),
                                action=torch.from_numpy(action[:idx]).to(torch.float32),
                                next_state=torch.from_numpy(next_state[:idx]).to(torch.float32),
                                reward=torch.from_numpy(reward[:idx]).to(torch.float32),
                                done=torch.from_numpy(done[:idx]).to(torch.float32))
            torch.save(episode_info, args.output_file_path)
    finally:
        env.close()