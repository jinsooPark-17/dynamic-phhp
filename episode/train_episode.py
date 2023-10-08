#!/usr/bin/env python3
from math import pi
import torch
from torch.distributions.uniform import Uniform

import argparse
from collections import namedtuple

from envs.simulation import I_Shaped_Hallway
from policy.policy import Actor
Pose = namedtuple("Pose", "x y yaw")

if __name__=="__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("storage", type=str, help="data storage directory")
    parser.add_argument("network_dir", type=str, help="Absolute directory of network")
    parser.add_argument("opponent", type=str, default="vanilla", help="Opponent robot type: [vanilla, baseline, custom, phhp, dynamic]")
    parser.add_argument("distance", type=float, default=10.0, help="Distance between two robots")
    parser.add_argument("frequency", type=float, default=1.0, help="Control frequency of policy")
    args = parser.parse_args()

    # Perform episode
    env = I_Shaped_Hallway()
    control_pi = Actor(n_scan=2)
    control_pi.load_state_dict( torch.load(args.network_dir) )

    # d = Uniform(low=4.0, high=12.0).sample().tolist()
    init_poses = [Pose(-12.0, 0.0, 0.0), Pose( -2.0, 0.0, -pi)]
    goal_poses = [Pose( -2.0, 0.0, 0.0), Pose(-12.0, 0.0, -pi)]
    s1, a, s2, r, d = env.run_episode(init_poses, goal_poses, args.opponent, timeout=60.0, 
                                      mode="explore", policy=control_pi, cycle = 0.8)

    torch.save(dict(state=s1, action=a, next_state=s2, reward=r, done=d), args.storage)
