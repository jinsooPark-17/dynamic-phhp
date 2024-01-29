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
    parser.add_argument("--network", type=torch.load, default=None)
    parser.add_argument("--n_scan", type=int, default=1, required=True, 
                        help="Defines the number of raw or hallucinated scans incorporated into the state space. (default: 1)")
    parser.add_argument("--sensor_horizon", type=float, default=8.0, required=True,
                        help="Specifies the maximum distance, in meters, for LiDAR sensor readings to be considered relevant; beyond this, data is ignored. (default: 8.0 m)")
    parser.add_argument("--plan_interval", type=float, required=True,
                        help="Configures the interval for plan quantization; the policy will utilize this quantized plan for decision-making. (default: 0.5 m)")
    parser.add_argument("--policy_hz", type=float, required=True,
                        help="Sets how often the HalAgent policy is applied each second, allowing for dynamic adjustment to system behavior. (default: 2.0 hz)")
    parser.add_argument("--opponent", type=str, choices=["vanilla", "baseline", "phhp"], required=True,
                        help="Select type of opponent robot that HalAgent must pass. [vanilla, baseline, phhp]")
    parser.add_argument("--combine_scans", action='store_true',
                        help="If activated, raw scan and hallucination scan will be combined to create features")
    parser.add_argument("--include_radius", action='store_true',
                        help="If activated, policy will determine (install, x, y, t, r) otherwise (install, x, y, t)")
    parser.add_argument("--test", action='store_true',
                        help="Activate when testing a trained policy.")
    parser.add_argument("--explore", action='store_true',
                        help="If, Activate the episode becomes pure exploration episode.")
    args = parser.parse_args()

    # Define constant
    N_PLAN = int(args.sensor_horizon/args.plan_interval*2)
    N_STATE = 2 * args.n_scan * 640 + N_PLAN + 2    # scan / plan / vw
    N_ACTION = (5 if args.include_radius else 4)
    MAX_SAMPLES = int(60.0 * args.policy_hz) + 1

    env = HallwayEpisode(
        num_scan_history=args.n_scan, 
        sensor_horizon=args.sensor_horizon, 
        plan_interval=args.plan_interval, 
        policy_hz=args.policy_hz
    )
    policy = ActorCritic(
        n_scan=args.n_scan,
        n_plan=N_PLAN,
        action_dim=N_ACTION,
        combine=args.combine_scans
    )
    if args.network_dir is not None:
        policy.load(args.network_dir)

    episode_info = dict(
        state      = np.zeros((MAX_SAMPLES, N_STATE), dtype=np.float32), 
        action     = np.zeros((MAX_SAMPLES, N_ACTION), dtype=np.float32),
        next_state = np.zeros((MAX_SAMPLES, N_STATE), dtype=np.float32),
        reward     = np.zeros(MAX_SAMPLES, dtype=np.float32),
        done       = np.zeros(MAX_SAMPLES, dtype=np.float32)
    )

    # Begin episode
    init_poses, goal_poses = generate_random_episode()
    state = env.reset(
        robot_modes=['vanilla', args.opponent],
        init_poses=init_poses, 
        goal_poses=goal_poses
    )
    done, idx = False, 0
    while not done:
        if args.explore is True:
            action = np.random.uniform(-1.0, 1.0, N_ACTION)
        else:
            action = policy.act(state, deterministic=args.test)
        next_state, reward, done = env.step(action)

        # Store episode data
        episode_info['state'][idx]      = state
        episode_info['action'][idx]     = action
        episode_info['next_state'][idx] = next_state
        episode_info['reward'][idx]     = reward
        episode_info['done'][idx]       = done

        state = next_state
        idx += 1
    
    # Store information as a file
    np.savez(args.output_file_path, 
             state=episode_info['state'][:idx],
             action=episode_info['action'][:idx],
             next_state=episode_info['next_state'][:idx],
             reward=episode_info['reward'][:idx],
             done=episode_info['done'][:idx])
    env.close()