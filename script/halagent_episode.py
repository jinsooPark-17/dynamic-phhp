import os
import time
import yaml
import argparse
import numpy as np
import torch
from environment import HallwayEpisode
from network import ActorCritic

def generate_random_episode():
    init_poses = np.zeros(2, 3)
    goal_poses = np.zeros(2, 3)

    d = np.random.uniform(  9., 16.)
    x = np.random.uniform(-22., -2.)

    init_poses[0] = [x, 0., 0.]
    if x+16 > 0:
        goal_poses[0] = [0., -(x+16), -np.pi/2.]
    else:
        goal_poses[0] = [x+16., 0., 0.]

    if x+d > 0.:
        init_poses[1] = [0., -(x+d), np.pi/2.]
    else:
        init_poses[1] = [x+d, 0, np.pi]
    goal_poses[1] = [x+d-16, 0, np.pi]
    return init_poses, goal_poses

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file_path", type=str, default='test', required=True)
    parser.add_argument("--network_dir", type=str, required=True)
    parser.add_argument("--config", type=str, help="configuration file *.yml", required=True)
    parser.add_argument("--command", type=str)
    parser.add_argument("--test", action='store_true',
                        help="Activate when testing a trained policy.")
    args = parser.parse_args()

    # Load configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Define constant
    N_PLAN = int(args.sensor_horizon/args.plan_interval * 2.)
    N_STATE = 2*config["policy"]["n_scan"]*640 + N_PLAN + 2        # scan / plan / vw
    MAX_SAMPLES = int(config["episode"]["timeout"] * args.policy_hz) + 1

    env = HallwayEpisode(
        num_scan_history=config["policy"]["n_scan"], 
        sensor_horizon=config["policy"]["sensor_horizon"], 
        plan_interval=config["policy"]["plan_interval"], 
        policy_hz=config["policy"]["policy_hz"]
    )
    policy = ActorCritic(
        n_scan=config["n_scan"],
        n_plan=N_PLAN,
        action_dim=config["act_dim"],
        combine=config["combine_scans"]
    )

    # Define data storage
    state      = np.zeros((MAX_SAMPLES, N_STATE)), 
    action     = np.zeros((MAX_SAMPLES, config["act_dim"])),
    next_state = np.zeros((MAX_SAMPLES, N_STATE)),
    reward     = np.zeros(MAX_SAMPLES),
    done       = np.zeros(MAX_SAMPLES)

    try:
        while True:
            # wait for the training process
            while os.path.exists(args.output_file_path):
                time.sleep(0.1)
            
            # Load new model
            policy.load_state_dict(torch.load(args.network_dir))

            # Begin episode
            init_poses, goal_poses = generate_random_episode()
            with open(os.path.join(args.commands, "opponent.command"), 'r') as f:
                opponent = f.read()

            state = env.reset(
                robot_modes=['vanilla', opponent],   # Get command from train proc
                init_poses=init_poses, 
                goal_poses=goal_poses
            )
            d, idx = False, 0
            while not d:
                if os.path.exists(os.path.join(args.commands, "explore.command")):
                    a = np.random.uniform(-1.0, 1.0, config["act_dim"])
                else:
                    s = torch.from_numpy(a).to(torch.float32)
                    a, _ = policy(s, deterministic=args.test)
                    a = a.numpy()
                ns, r, d = env.step(a)

                # Store SAS'RD
                state[idx]      = s
                action[idx]     = a
                next_state[idx] = ns
                reward[idx]     = r
                done[idx]       = d

                # Update
                s = ns
                idx = idx + 1
            
            # Store episode information as a file
            episode_info = dict(state=torch.from_numpy(state[:idx]).to(torch.float32),
                                action=torch.from_numpy(action[:idx]).to(torch.float32),
                                next_state=torch.from_numpy(next_state[:idx]).to(torch.float32),
                                reward=torch.from_numpy(reward[:idx]).to(torch.float32),
                                done=torch.from_numpy(done[:idx]).to(torch.float32))
            torch.save(episode_info, args.output_file_path)
    finally:
        env.close()