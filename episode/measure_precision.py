#!/usr/bin/env python3
import numpy as np
import argparse
from collections import namedtuple
from envs.simulation import I_Shaped_Hallway
Pose = namedtuple("Pose", "x y yaw")

if __name__=="__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("storage", type=str, help="Name of singularity container instance")
    parser.add_argument("num_test", type=int, help="Number of episodes")
    args = parser.parse_args()

    # Perform episode
    env = I_Shaped_Hallway()

    result = np.empty((2*args.num_test,4))
    for i in range(args.num_test):
        d1, d2 = np.random.uniform(low=6.0, high=10.0, size=2)
        init_poses = [Pose(-10.0   , 0.0, 0.0), Pose(0.0, 0.0   , -np.pi/2.)]
        goal_poses = [Pose(-10.0+d1, 0.0, 0.0), Pose(0.0, 0.0-d2, -np.pi/2.)]
        ttd = env.test_precision(init_poses, goal_poses, timeout=40.0)
        result[2*i:2*i+2] = np.hstack(([[d1],[d2]], ttd))

    # save numpy array to file
    with open(args.storage, 'w') as f:
        np.save(f, result)
