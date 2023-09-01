#/usr/bin/env python3.8
from collections import namedtuple
from envs.robots import Vanilla, Gazebo
import numpy as np
import rospy

Pose = namedtuple("Pose", "x y yaw")

class L_Hallway_Single_robot(Gazebo):
    def __init__(self, ep_timeout: float=60.0):
        super().__init__()
        self.robot1 = None
        self.timeout = ep_timeout
        self.rate = rospy.Rate(10)

    def register_robots(self, robot1):
        self.robot1 = robot1

    def begin(self, init_poses, goal_poses):
        # Reset episode
        ## stop robot
        self.robot1.stop()
        ## teleport robots
        self.teleport(self.robot1.id, init_poses[0].x, init_poses[0].y, init_poses[0].yaw)
        rospy.sleep(1.0)
        ## localize robots
        self.robot1.localize(init_poses[0].x, init_poses[0].y, init_pose[0].yaw)
        rospy.sleep(1.0)
        for _ in range(10):
            self.robot1.clear_costmap()
            rospy.sleep(0.1)

        # Move robots to goal pose
        self.robot1.goto(goal_poses[0].x, goal_poses[0].y, goal_poses[0].yaw, timeout=self.timeout)

        # wait for episode to finish
        while not rospy.is_shutdown():
            if self.robot1.is_arrived():
                break
            self.rate.sleep()

        return self.robot1.trajectory

if __name__ == "__main__":
    name, init_pose = np.random.choice([["marvin", Pose(-10,0,0)], ["rob", Pose(0, -10, np.pi/2)]])
    env = L_Hallway_Single_robot(ep_timeout=30.0)
    env.register_robots(robot1=Vanilla( name ))

    trajectory = env.begin()

    print(trajectory)