#/usr/bin/env python3.8
from collections import namedtuple
from envs.robots import Vanilla
from envs.simulation import L_Hallway_Single_robot
import numpy as np

Pose = namedtuple("Pose", "x y yaw")

if __name__ == "__main__":
    idx = np.random.choice(2)
    name, init_pose = [["marvin", Pose(-10,0,0)], ["rob", Pose(0, -10, np.pi/2.)]][idx]
    env = L_Hallway_Single_robot(ep_timeout=30.0)
    env.register_robots(robot1=Vanilla( name ))

    trajectory = env.begin([init_pose], [Pose(0, 0, np.pi/4.)])

    print(trajectory)