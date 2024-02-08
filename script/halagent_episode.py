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
    
    center   = np.random.uniform(-10.0, 10.0)
    d_robots = np.random.uniform(3.0, 8.0)
    d_goal   = 10.0
    
    if center-d_robots < 0.:
        init_poses[0] = [center-d_robots, 0., 0.]
    else:
        init_poses[0] = [0., -(center-d_robots), -np.pi/2.]
    
    if center+d_robots < 0.:
        init_poses[1] = [center + d_robots, 0., np.pi]
    else:
        init_poses[1] = [0., -(center+d_robots), np.pi/2.]

    if center-d_robots+d_goal < 0.:
        goal_poses[0] = [center-d_robots+d_goal, 0., 0.]
    else:
        goal_poses[0] = [0., -(center-d_robots+d_goal), -np.pi/2.]
    
    if center+d_robots-d_goal < 0.:
        goal_poses[1] = [center+d_robots-d_goal, 0., np.pi]
    else:
        goal_poses[1] = [0., -(center+d_robots-d_goal), np.pi/2.]

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