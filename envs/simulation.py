import os
import time
import numpy as np
from math import sin, cos

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class Gazebo:
    def __init__(self, debug: str=""):
        start_time = time.time()
        rospy.init_node("environment", anonymous=True)
        if debug:
            ros_t = rospy.Time.now().to_sec()
            print(f"{debug}: rospy.init_node took {time.time() - start_time:.3f} sec", flush=True)

        start_time = time.time()
        while not rospy.is_shutdown() and not (60.0 < rospy.Time.now().to_sec() < 1e+6):
            time.sleep(0.1)
        if debug:
            print(f"{debug}: wait for (ROS_Time {ros_t:.3f} -> {rospy.Time.now().to_sec():.3f}) took {time.time() - start_time:.3f} sec", flush=True)

        self.teleport_srv = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

    def teleport(self, entity_name, x, y, yaw):
        rospy.wait_for_service("/gazebo/set_model_state", timeout=1.0)
        try:
            req = ModelState(model_name=entity_name)
            req.reference_frame = "world"
            req.pose.position.x = x
            req.pose.position.y = y
            req.pose.orientation.z = sin(yaw/2.)
            req.pose.orientation.w = cos(yaw/2.)
            self.teleport_srv(req)
            rospy.sleep(1.0)
        except (rospy.ServiceException) as e:
            raise RuntimeError("Gazebo: set_model_state does not respond.")

class L_Hallway_Single_robot(Gazebo):
    def __init__(self, ep_timeout: float=60.0, debug: str=""):
        super().__init__(debug)
        start_time = time.time()
        self.robot1 = None
        self.timeout = ep_timeout
        self.rate = rospy.Rate(100)

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
        self.robot1.localize(init_poses[0].x, init_poses[0].y, init_poses[0].yaw)
        rospy.sleep(1.0)
        for _ in range(10):
            self.robot1.clear_costmap()
            self.robot1.clear_hallucination()
            rospy.sleep(0.1)

        # Move robots to goal pose
        start_time = rospy.Time.now()
        self.robot1.move(goal_poses[0].x, goal_poses[0].y, goal_poses[0].yaw, timeout=self.timeout)

        # wait for episode to finish
        while not rospy.is_shutdown():
            if not self.robot1.is_running():
                break
            self.rate.sleep()

        return self.robot1.ttd # self.robot1.trajectory

class I_Hallway(Gazebo):
    def __init__(self, robot1=None, robot2=None, ep_timeout: float=60.0, hz: float=1.0):
        super().__init__()
        self.robot1 = robot1
        self.robot2 = robot2
        self.timeout = ep_timeout
        self.rate = rospy.Rate(hz)
        self.max_len = int((ep_timeout + 1.0) * hz)

    def register_robots(self, robot1, robot2):
        self.robot1 = robot1
        self.robot2 = robot2

    def execute(self, init_poses, goal_poses, storage: str=None):
        # Prepare data storage
        state  = np.zeros(shape=(self.max_len+1, 640+2+20), dtype=np.float32)
        action = np.zeros(shape=(self.max_len, 3), dtype=np.float32)
        reward = np.zeros(shape=(self.max_len,), dtype=np.float32)
        done   = np.zeros(shape=(self.max_len,), dtype=np.float32)

        explore=False
        if os.path.exists("explore.command"):
            explore=True

        # stop robots
        self.robot1.stop()
        self.robot2.stop()
        rospy.sleep(1.0)

        # Reset robots to initial pose
        self.teleport(self.robot1.id, init_poses[0].x, init_poses[0].y, init_poses[0].yaw)
        self.teleport(self.robot2.id, init_poses[1].x, init_poses[1].y, init_poses[1].yaw)
        rospy.sleep(1.0)

        self.robot1.localize(init_poses[0].x, init_poses[0].y, init_poses[0].yaw)
        self.robot2.localize(init_poses[1].x, init_poses[1].y, init_poses[1].yaw)
        rospy.sleep(1.0)

        self.robot1.reset(init_poses[0].x, init_poses[0].y, init_poses[0].yaw)
        self.robot2.reset(init_poses[1].x, init_poses[1].y, init_poses[1].yaw)
        rospy.sleep(1.0)

        # Move robots to goal pose
        self.robot1.goto(goal_poses[0].x, goal_poses[0].y, goal_poses[0].yaw, timeout=self.timeout)
        self.robot2.goto(goal_poses[1].x, goal_poses[1].y, goal_poses[1].yaw, timeout=self.timeout)

        # wait for episode to finished
        idx = 0
        while not rospy.is_shutdown():
            # terminate episode when both robot finished
            if self.robot1.is_arrived() and self.robot2.is_arrived():
                state[idx], _, _ = self.robot1.hallucinate()
                reward[idx-1] += (30. if self.robot1.success else 0.) # Add success incentive
                done[idx-1] = True

            if not self.robot1.is_arrived():
                state[idx], action[idx], reward[idx] = self.robot1.hallucinate(explore=explore)
                idx += 1

            if not self.robot2.is_arrived():
                self.robot2.hallucinate(explore=explore)

            self.rate.sleep()

        traj1 = self.robot1.trajectory[:self.robot1.idx]
        traj2 = self.robot2.trajectory[:self.robot2.idx]

        if storage is not None:
            if (np.linalg.norm(traj1[-1,:2] - traj1[0,:2]) < 0.5) or (np.linalg.norm(traj2[-1,:2] - traj2[0,:2]) < 0.5):
                return

            with open(storage, 'wb') as f:
                np.savez(
                    f, obs1=state[:idx], action=action[:idx], obs2=state[1:idx+1],
                    reward=reward[:idx], done=done[:idx],
                    robot1=traj1, robot2=traj2
                )
