import os
import time
import yaml
import argparse
import numpy as np
import torch
from environment import HallwayEpisode
from network import Actor

def generate_random_episode():
    init_poses = np.zeros((2, 3))
    goal_poses = np.zeros((2, 3))

    g = 10.0                            # Distance between init_pose to goal_pose (before: 16.0)
    d = np.random.uniform(  6., 16.)    # Distance between robot_1 and robot_2
    x = np.random.uniform(-17., -1.)    # robot_1's initial x location

    init_poses[0] = [x, 0., 0.]
    if x+16 > 0:
        goal_poses[0] = [0., -(x+g), -np.pi/2.]
    else:
        goal_poses[0] = [x+g, 0., 0.]

    if x+d > 0.:
        init_poses[1] = [0., -(x+d), np.pi/2.]
    else:
        init_poses[1] = [x+d, 0, np.pi]
    goal_poses[1] = [x+d-g, 0, np.pi]
    return init_poses, goal_poses

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file_path", type=str, default='test')
    parser.add_argument("--network_dir", type=str, required=True)
    parser.add_argument("--config", type=str, help="configuration file *.yml", required=True)
    parser.add_argument("--commands", type=str)
    parser.add_argument("--test", action='store_true',
                        help="Activate when testing a trained policy.")
    args = parser.parse_args()
    deterministic = args.test
    result_file = args.output_file_path

    # Load configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Define constant
    N_PLAN = int(config["policy"]["sensor_horizon"]/config["policy"]["plan_interval"] * 2.)
    N_STATE = 2*config["policy"]["n_scan"]*640 + N_PLAN + 2        # scan / plan / vw
    MAX_SAMPLES = int(config["episode"]["timeout"] * config["policy"]["policy_hz"]) + 10

    env = HallwayEpisode(
        num_scan_history=config["policy"]["n_scan"], 
        sensor_horizon=config["policy"]["sensor_horizon"], 
        plan_interval=config["policy"]["plan_interval"], 
        policy_hz=config["policy"]["policy_hz"],
        c_plan_change=config["reward"]["C_PLAN_CHANGE"],
        c_stop=config["reward"]["C_STOP"],
        c_success=config["reward"]["C_SUCCESS"]
    )
    policy = Actor(
        n_scan=config["policy"]["n_scan"],
        n_plan=N_PLAN,
        action_dim=config["policy"]["act_dim"],
        combine=config["policy"]["combine_scans"]
    )

    # Define data storage
    state      = np.zeros((MAX_SAMPLES, N_STATE))
    action     = np.zeros((MAX_SAMPLES, config["policy"]["act_dim"]))
    next_state = np.zeros((MAX_SAMPLES, N_STATE))
    reward     = np.zeros(MAX_SAMPLES)
    done       = np.zeros(MAX_SAMPLES)

    try:
        while True:
            # wait for the training process
            while os.path.exists(result_file):
                if os.path.exists(os.path.join(args.commands, "test.command")):
                    deterministic = True
                    result_file = args.output_file_path.replace('.', '.test.')
                    break
                time.sleep(0.1)
            else:
                deterministic = args.test
            
            # Load new model
            policy.load_state_dict(torch.load(args.network_dir))

            # Begin episode
            init_poses, goal_poses = generate_random_episode()
            with open(os.path.join(args.commands, "opponent.command"), 'r') as f:
                opponent = f.read()

            s = env.reset(robot_modes=['vanilla', opponent],   # Get command from train proc
                          init_poses=init_poses, 
                          goal_poses=goal_poses)
            d, idx = False, 0
            while not d:
                if os.path.exists(os.path.join(args.commands, "explore.command")):
                    a = np.random.uniform(-1.0, 1.0, config["policy"]["act_dim"])
                else:
                    with torch.no_grad():
                        a, _ = policy(torch.from_numpy(s).to(torch.float32), deterministic=deterministic)
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
            torch.save(episode_info, result_file)
            result_file = args.output_file_path
            env.close()
    finally:
        env.close()