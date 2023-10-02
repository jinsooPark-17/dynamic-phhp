#!/usr/bin/env python3
import numpy as np
import argparse
from collections import namedtuple
from envs.simulation import I_Shaped_Hallway
Pose = namedtuple("Pose", "x y yaw")

if __name__=="__main__":
    import rospy
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("storage", type=str, help="Name of singularity container instance")
    args = parser.parse_args()

    # Perform episode
    env = I_Shaped_Hallway()

    d1, d2 = np.random.uniform(low=6.0, high=10.0, size=2)
    init_poses = [Pose(-10.0   , 0.0, 0.0), Pose(0.0, 0.0, -np.pi)]
    goal_poses = [Pose(-10.0+d1, 0.0, 0.0), Pose(-d2, 0.0, -np.pi)]

    # randomize init & goal position
    if np.random.normal() > 0.0:
        init_poses = init_poses[::-1]
        goal_poses = goal_poses[::-1]
    env.reset(init_poses)

    env.robot1.move(goal_poses[0].x, goal_poses[0].y, goal_poses[0].yaw, mode="baseline", timeout=60.0, detection_range=9.5, comms_topic="/rob/amcl_pose")
    env.robot2.move(goal_poses[1].x, goal_poses[1].y, goal_poses[1].yaw, mode="phhp", timeout=60.0, detection_range=6.5, comms_topic="/marvin/amcl_pose")

    while env.robot1.is_running() or env.robot2.is_running():
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException("ROS shutdown while running episode")
        rospy.sleep(0.1)

    # return ttd and percision error (r, th)
    ttd1 = env.robot1.ttd
    success1 = env.robot1.is_arrived()
    last_loc1 = env.robot1.trajectory[env.robot1.traj_idx-1]
    dx1, dy1 = goal_poses[0].x - last_loc1[0], goal_poses[0].y - last_loc1[1]

    ttd2 = env.robot2.ttd
    success2 = env.robot2.is_arrived()
    last_loc2 = env.robot2.trajectory[env.robot2.traj_idx-1]
    dx2, dy2 = goal_poses[1].x - last_loc2[0], goal_poses[1].y - last_loc2[1]

    ep_result = np.array([[ttd1, dx1, dy1, success1], [ttd2, dx2, dy2, success2]])
    print(ep_result)
