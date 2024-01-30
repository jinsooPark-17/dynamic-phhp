import argparse
import torch
import numpy as np
import gymnasium as gym
from network import MLPActor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file_path", type=str, default="test")
    parser.add_argument("--network_dir", type=torch.load, required=True)
    parser.add_argument("--explore", action='store_true')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    if args.debug:
        env = gym.make("Pendulum-v1", render_mode='human')
    else:
        env = gym.make("Pendulum-v1")
    policy = MLPActor(n_obs=3, n_act=1, hidden=(128,128,))
    policy.load_state_dict(args.network_dir)

    # Define data storage
    episode_info = dict(
        state = np.zeros((200,3)),
        action = np.zeros((200,1)),
        next_state = np.zeros((200,3)),
        reward = np.zeros(200,),
        done = np.zeros(200,)
    )

    (state, _), done, idx = env.reset(), False, 0
    while not done:
        if args.explore:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(torch.float32)
                action, _ = policy(state)
                action = action.numpy()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = (terminated or truncated)

        # Store SAS'RD
        episode_info['state'][idx]      = state
        episode_info['action'][idx]     = action
        episode_info['next_state'][idx] = next_state
        episode_info['reward'][idx]     = reward
        episode_info['done'][idx]       = done

        # Update
        state = next_state
        idx += 1

    # Store episode info as a file
    np.savez(args.output_file_path,
             state=episode_info['state'][:idx],
             action=episode_info['action'][:idx],
             next_state=episode_info['next_state'][:idx],
             reward=episode_info['reward'][:idx],
             done=episode_info['done'][:idx])
    env.close()