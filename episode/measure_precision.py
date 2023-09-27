#!/usr/bin/env python3
import os
import numpy as np
import rospy
import argparse
from collections import namedtuple
from envs.simulation import I_Shaped_Hallway
Pose = namedtuple("Pose", "x y yaw")

if __name__=="__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("storage", type=str, help="Name of singularity container instance")
    args = parser.parse_args()

    # Perform episode
    env = I_Shaped_Hallway()
    d1, d2 = np.random.uniform(low=6.0, high=10.0, size=2)
    init_poses = [Pose(-10.0   , 0.0, 0.0), Pose(0.0, 0.0   , -np.pi/2.)]
    goal_poses = [Pose(-10.0+d1, 0.0, 0.0), Pose(0.0, 0.0-d2, -np.pi/2.)]
    result = env.test_precision(init_poses, goal_poses, timeout=40.0)
    result = np.hstack(([[d1],[d2]], result))

    # save numpy array to file
    with open(args.storage, 'ab') as f:
        np.save(f, result)
