import numpy as np
from math import sin, cos

import rospy
from agent import Agent, quaternion_to_yaw
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class GazeboController(object):
    def __init__(self):
        rospy.init_node("episode_handler_py", anonymous=True)
        rospy.sleep(1.0)
        # Define service
        self.__teleport_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.__pause_srv    = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.__unpause_srv  = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
    def teleport(self, entity_name, x, y, yaw):
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            req = ModelState(model_name=entity_name)
            req.reference_frame="world"
            req.pose.position.x = x
            req.pose.position.y = y
            req.pose.orientation.z = sin(yaw/2.)
            req.pose.orientation.w = cos(yaw/2.)
            self.__teleport_srv(req)
        except rospy.ServiceException as e:
            raise RuntimeError("Gazebo: teleport service failed.")
    def pause(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.__pause_srv()
        except rospy.ServiceException as e:
            raise RuntimeError("/gazebo/pause_physics does not respond.")
    def unpause(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.__unpause_srv()
        except rospy.ServiceException as e:
            raise RuntimeError("/gazebo/unpause_physics does not respond.")
    def reset(self):
        raise NotImplementedError()
    def step(self):
        raise NotImplementedError()

class HallwayEpisode(GazeboController):
    def __init__(self, num_scan_history=1, sensor_horizon=8.0, plan_interval=0.5, policy_hz=2, c_plan_change=1.0, c_stop=0.5, c_success=10.0):
        super().__init__()
        self.robots = [
            Agent(id = 'marvin', 
                  num_scan_history = num_scan_history, 
                  sensor_horizon   = sensor_horizon, 
                  plan_interval    = plan_interval),
            Agent(id = 'rob', 
                  num_scan_history = num_scan_history, 
                  sensor_horizon   = sensor_horizon, 
                  plan_interval    = plan_interval),
        ]

        # Define reward constancts
        self.C_PLAN_CHANGE = c_plan_change
        self.C_STOP = c_stop
        self.C_SUCCESS = c_success

        # Define episode control parameters
        self.compute_deply_correction = rospy.Rate(10.)
        self.episode_duration = rospy.Rate(policy_hz)

    def reset(self, robot_modes, init_poses: list, goal_poses: list):
        while not rospy.is_shutdown():
            # cancel all previous episodes
            self.unpause()
            for _ in range(10):
                for robot in self.robots:
                    robot.stop()
                    robot.clear_hallucination()
                rospy.sleep(0.1)
            rospy.sleep(0.5)

            # Define new episode parameters
            traffic = np.random.choice(['left', 'right'])
            mirror  = np.random.choice([True, False])
            if mirror:
                init_poses = init_poses[::-1]
                goal_poses = goal_poses[::-1]
            comms_topics = [f'{robot.id}/amcl_pose' for robot in self.robots[::-1]]

            # teleport robot to init_pose
            for _ in range(5):
                for robot, init_pose in zip(self.robots, init_poses):
                    self.teleport(robot.id, *init_pose) # init_pose := (x,y,yaw)
                rospy.sleep(0.1)
            rospy.sleep(0.5)

            # Localize robot
            for _ in range(10):
                for robot, init_pose in zip(self.robots, init_poses):
                    robot.localize(*init_pose)
            rospy.sleep(1.0)

            # Clear costmap and hallucinations
            for _ in range(10):
                for robot in self.robots:
                    robot.clear_costmap()
                    robot.clear_hallucination()
                rospy.sleep(0.1)
            rospy.sleep(1.0)

            # Begin episode
            self.episode_duration.sleep()   # wait for start line
            for robot, comms_topic, mode, goal_pose in zip(self.robots, comms_topics, robot_modes, goal_poses):
                robot.move(*goal_pose, timeout=60., mode=mode, traffic=traffic, comms_topic=comms_topic)
            self.episode_duration.sleep()
            self.pause()

            # Check if robots actually moves
            for robot, (x, y, yaw) in zip(self.robots, init_poses):
                if (robot.pose.position.x - x)**2 + (robot.pose.position.y - y)**2 < 0.05 and abs(quaternion_to_yaw(robot.pose.orientation) - yaw) < 0.3:
                    print(f"DEBUG: {robot.id} does not move! go back to reset process.")
                    break
            else:
                obs   = self.robots[0].get_state()
                state = np.concatenate((obs['scan'], obs['plan'], obs['vw']), axis=None)
                return state

    def step(self, action):
        self.unpause()
        self.compute_deply_correction.sleep()

        # Do SOMETHING
        self.robots[0].action(*action)

        # Finish action
        self.episode_duration.sleep()
        self.pause()

        obs    = self.robots[0].get_state()
        done   = (not self.robots[0].is_running())
        reward = -self.C_PLAN_CHANGE * obs['hausdorff_dist'] - self.C_STOP * obs['penalty']
        if self.robots[0].is_arrived():
            reward += self.C_SUCCESS - self.robots[0].ttd
        state  = np.concatenate((obs['scan'], obs['plan'], obs['vw']), axis=None)
        return state, reward, done

    def close(self):
        self.unpause()

        for robot in self.robots:
            robot.stop()
        rospy.sleep(0.1)

        for robot in self.robots:
            robot.clear_costmap()
            robot.clear_hallucination()

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode1",     type=str, choices=["vanilla", "baseline", "phhp", "dynamic"], required=True)
    # parser.add_argument("--mode2",     type=str, choices=["vanilla", "baseline", "phhp", "dynamic"], required=True)
    # parser.add_argument("--n_episode", type=int, required=True)
    # args = parser.parse_args()

    # env = HallwayEpisode(
    #     num_scan_history=1, 
    #     sensor_horizon=8.0, 
    #     plan_interval=0.5
    # )

    # init_poses = [[-16.0, 0., 0.],
    #               [-6.0, 0., np.pi]]
    # goal_poses = [[-0., 0., 0.],
    #               [-22.0, 0., np.pi]]
    # print(f"Episode reward of {args.mode1}-{args.mode2} episodes")
    # for n in range(args.n_episode):
    #     obs, done = env.reset(robot_modes=[args.mode1, args.mode2], init_poses=init_poses, goal_poses=goal_poses), False
    #     reward_hist = []

    #     while not done:
    #         act = np.random.rand(3)
    #         obs, rew, done = env.step(act)
    #         reward_hist += [rew]

    #     print(f"\tEpisode {n+1}:\n\t\treward: {sum(reward_hist)}\n\t\tTTD: {env.robots[0].ttd}")
    #     plt.plot(reward_hist[:-1])
    # plt.savefig(f"{args.mode1}_{args.mode2}.png")

    """ Test 2: check agent action """
    # import matplotlib.pyplot as plt
    # env = HallwayEpisode(
    #     num_scan_history=1,
    #     sensor_horizon=8.0,
    #     plan_interval=0.5
    # )

    # init_poses = [[-16.0, 0., 0.],
    #               [-6.0, 0., np.pi]]
    # goal_poses = [[-0., 0., 0.],
    #               [-22.0, 0., np.pi]]

    # for i, action in enumerate([[-0.1, 0.5, 0.0, 0.0],[0.5, 0.5, 0.0, 0.0],[0.5, 0.5, 0.2, 0.0],[0.5, 0.5, -0.2, 0.0]]):
    #     if i==0:
    #         msg="Do not install"
    #     elif i==1:
    #         msg="Install VO 6 meters ahead that blocks center"
    #     elif i==2:
    #         msg="Install VO 6 meters ahead that blocks right"
    #     elif i==3:
    #         msg="Install VO 6 meters ahead that blocks left"
    #     print(f"CASE {i+1}: {msg}")

    #     obs, done = env.reset(robot_modes=['vanilla', 'vanilla'], init_poses=init_poses, goal_poses=goal_poses), False
    #     obs, rew, done = env.step(action)
    #     rospy.sleep(0.001)
    #     for n in range(9):
    #         obs, rew, done = env.step([-1.0, -1.0, -1.0, -1.0])
    #         scan = obs['scan']
    #         ego_plan = obs['plan']

    #         plt.figure()
    #         plt.scatter(scan[0]*np.cos(env.robots[0].theta), scan[0]*np.sin(env.robots[0].theta), c='r', alpha=0.5, s=1)
    #         plt.scatter(scan[1]*np.cos(env.robots[0].theta), scan[1]*np.sin(env.robots[0].theta), c='b', alpha=0.5, s=1)
    #         plt.plot(ego_plan[:,0]*np.cos(ego_plan[:,1]), ego_plan[:,0]*np.sin(ego_plan[:,1]), c='k')
    #         plt.title(f"v: {obs['vw'][0]:.2f}, w: {obs['vw'][1]:.2f}")
    #         plt.savefig("agent_state.png")
    #         plt.close()
    #         rospy.sleep(0.001)
    #     print("Episode done")

    """ Test 3: check global costmap """
    import matplotlib.pyplot as plt
    env1 = HallwayEpisode(
        num_scan_history=1,
        sensor_horizon=8.0,
        plan_interval=0.5
    )

    init_poses = [[-16.0, 0., 0.],
                  [-6.0, 0., np.pi]]
    goal_poses = [[-0., 0., 0.],
                  [-22.0, 0., np.pi]]

    s = env1.reset(robot_modes=['vanilla', 'vanilla'], init_poses=init_poses, goal_poses=goal_poses)
    done = False
    while not done:
        ns, r, done = env1.step([-1.0, -1.0, -1.0, -1.0])
        s = ns
    
    env2 = HallwayEpisode(
        num_scan_history=1,
        sensor_horizon=8.0,
        plan_interval=0.5
    )

    print(np.all(env1.robots[0].map.costmap == env2.robots[0].map.costmap))
    print(env1.robots[0].map.costmap == env2.robots[0].map.costmap)