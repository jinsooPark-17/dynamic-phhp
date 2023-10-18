#!/usr/bin/env python3
from math import pi
import torch

import argparse
from collections import namedtuple

from envs.simulation import I_Shaped_Hallway
from policy.policy import Actor
Pose = namedtuple("Pose", "x y yaw")

if __name__=="__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=str, required=True, help="Absolute path of file where resulting (S, A, S', R, D) is stored.")
    parser.add_argument("--network", type=torch.load, required=True, help="Absolute path of pyTorch network file.")
    parser.add_argument("--mode", type=str, choices=["explore", "exploit", "evaluate"], required=True)
    parser.add_argument("--opponent", type=str, choices=['vanilla', 'baseline', 'custom', 'phhp', 'dynamic'], required=True, help="Choose opponent behavior")
    parser.add_argument("--init_poses", type=float, nargs=3, action='append', metavar=('x','y','yaw'))
    parser.add_argument("--goal_poses", type=float, nargs=3, action='append', metavar=('x','y','yaw'))
    parser.add_argument("--timeout", type=float, default=60.0, help="Set timeout for episode (default: 60.0s)")
    parser.add_argument("--hz", type=float, default=1.0, help="Set frequency of RL-policy (default: 1.0/s)")
    args = parser.parse_args()

    # assign default values to init_poses and goal_poses
    if args.init_poses == args.goal_poses == None:
        args.init_poses = [Pose(-12.0, 0.0, 0.0), Pose(-2.0, 0.0, pi)]
        args.goal_poses = [Pose(-2.0, 0.0, 0.0), Pose(-12.0, 0.0, pi)]
    else:
        args.init_poses = [Pose(*p) for p in args.init_poses]
        args.goal_poses = [Pose(*p) for p in args.goal_poses]
    assert len(args.init_poses) == len(args.goal_poses) == 2

    # Run episode
    env = I_Shaped_Hallway()
    model = Actor(n_scan=2)
    model.load_state_dict(args.network)
    s1, a, s2, r, d = env.run_episode(
        init_poses=args.init_poses, goal_poses=args.goal_poses, opponent=args.opponent, timeout=args.timeout,
        mode=args.mode, policy=model, cycle=args.hz, shuffle=(False if args.mode=='evaluate' else True)
    )
    torch.save(dict(state=s1, action=a, next_state=s2, reward=r, done=d, 
                    trajectory1=torch.from_numpy(env.robot1.get_trajectory()).to(torch.float32), 
                    trajectory2=torch.from_numpy(env.robot2.get_trajectory()).to(torch.float32)), 
               args.storage)
