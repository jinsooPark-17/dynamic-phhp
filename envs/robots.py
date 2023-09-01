#/usr/bin/env python3
import os
import torch
import numpy as np
from copy import deepcopy
from math import sin, cos, atan2

import rospy
from actionlib import SimpleActionClient
from std_srvs.srv import Empty
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from geometry_msgs.msg import PolygonStamped, Point32

class MoveBase(object):
    def __init__(self, id: str=""):
        self.id = id
        self.goal = MoveBaseGoal()

        # define ROS move_base
        self.move_base = SimpleActionClient( os.path.join(self.id, "move_base"), MoveBaseAction )
        connected = self.move_base.wait_for_server(timeout=rospy.Duration(30.0))
        if not connected:
            raise TimeoutError("MoveBase not respond")
        # define clear costmap service
        self.__clear_costmap_srv = rospy.ServiceProxy(os.path.join(self.id, "move_base", "clear_costmaps"), Empty)

    def goto(self, x: float, y: float, yaw: float, timeout: int=60):
        self.goal.target_pose.header.frame_id = os.path.join(self.id, "level_mux_map")
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.goal.target_pose.pose.position.x = x
        self.goal.target_pose.pose.position.y = y
        self.goal.target_pose.pose.orientation.z = sin(yaw/2.)
        self.goal.target_pose.pose.orientation.w = cos(yaw/2.)

        self.move_base.send_goal(
            goal        = self.goal,
            active_cb   = self.active_cb,
            feedback_cb = self.feedback_cb,
            done_cb     = self.done_cb
        )

    def clear_costmap(self):
        rospy.wait_for_service(os.path.join(self.id, "move_base", "clear_costmaps"))
        try:
            self.__clear_costmap_srv()
        except rospy.ServiceException as e:
            raise RuntimeError(e)

    def stop(self):
        self.move_base.cancel_all_goals()

    def active_cb(self):
        raise NotImplementedError("No active callback")
    
    def feedback_cb(self, feedback):
        raise NotImplementedError("No feedback callback")
    
    def done_cb(self, state, result):
        raise NotImplementedError("No done callback")
    
    def is_arrived(self):
        """ Pending = 0, Active = 1, Done = 2 """
        return self.move_base.simple_state == 2

class Vanilla(MoveBase):
    def __init__(self, id):
        super().__init__(id)
        self.ttd = None
        self.success = False
        self.curr_pose = None
        self.trajectory, self.idx = np.zeros(shape=(0,3), dtype=np.float32), 0

        # localize AMCL
        self.pub_localize = rospy.Publisher(os.path.join(self.id, "initialpose"), PoseWithCovarianceStamped, queue_size=10)

    def goto(self, x: float, y: float, yaw: float, timeout: float=60.0):
        # store 20hz feedback
        self.idx = 0
        self.trajectory = np.zeros(shape=(int(20*(timeout+1)), 3), dtype=np.float32)

        return super().goto(x, y, yaw, timeout)

    def localize(self, x: float, y: float, yaw: float, var: float=0.01):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = os.path.join(self.id, "level_mux_map")
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = sin(yaw/2.)
        msg.pose.pose.orientation.w = cos(yaw/2.)
        msg.pose.covariance = [
            #X    Y    Z    R    P    Y
            var, 0.0, 0.0, 0.0, 0.0, 0.0,   # X
            0.0, var, 0.0, 0.0, 0.0, 0.0,   # Y
            0.0, 0.0, var, 0.0, 0.0, 0.0,   # Z
            0.0, 0.0, 0.0, var, 0.0, 0.0,   # R
            0.0, 0.0, 0.0, 0.0, var, 0.0,   # P
            0.0, 0.0, 0.0, 0.0, 0.0, var,   # Y
        ]

        self.curr_pose = msg.pose.pose

        self.pub_localize.publish(msg)

    def reset(self, x:float, y: float, yaw: float, var: float=0.01):
        self.localize(x, y, yaw, var)
        self.clear_costmap()

    # Define behavior of robot
    def active_cb(self):
        self.success = False
        self.ttd = rospy.Time.now()

    def feedback_cb(self, feedback):
        self.curr_pose = feedback.base_position.pose
        x = self.curr_pose.position.x
        y = self.curr_pose.position.y
        q = self.curr_pose.orientation
        yaw = atan2(2.0 * q.z * q.w, 1.0 - 2.0 * q.z * q.z)
        self.trajectory[self.idx] = [x, y, yaw]
        self.idx += 1

        # cancel goal if move_base failed to arrive goal location by timeout
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()
    
    def done_cb(self, state, result):
        self.clear_costmap()
        self.ttd = (rospy.Time.now() - self.ttd).to_sec()
        self.success = (state == 3) # True only if state is SUCCEEDED
        self.trajectory = self.trajectory[:self.idx]

    def hallucinate(self, **kwargs):
        """ Dummy function for simplicity """
        return None, None, None

class Baseline(Vanilla):
    def __init__(self, id):
        super().__init__(id)

        # define perceptual hallucination
        self.ROBOT_RADIUS = 0.4

        self.make_plan_srv           = rospy.ServiceProxy(os.path.join(self.id, "move_base", "NavfnROS", "make_plan"), GetPlan)
        self.clear_hallucination_srv = rospy.ServiceProxy(os.path.join(self.id, "clear_virtual_circles"), Empty)
        self.pub_hallucination       = rospy.Publisher(os.path.join(self.id, "add_circles"), PolygonStamped, queue_size=10)

    def clear_hallucination(self):
        rospy.wait_for_service(os.path.join(self.id, "clear_virtual_circles"))
        try:
            self.clear_hallucination_srv()
        except rospy.ServiceException as e:
            raise RuntimeError(e)

    def reset(self, x: float, y: float, yaw: float, var: float = 0.01):
        self.clear_hallucination()
        return super().reset(x, y, yaw, var)

    def active_cb(self):
        RADIUS = 0.5
        # Get plan from move_base
        plan_req = GetPlanRequest()
        plan_req.start.header.frame_id = plan_req.goal.header.frame_id = os.path.join(self.id, "level_mux_map")
        plan_req.start.pose = self.curr_pose
        plan_req.goal.pose = self.goal.target_pose.pose

        rospy.wait_for_service(os.path.join(self.id, "move_base", "make_plan"))
        try:
            plan_msg = self.make_plan_srv( plan_req )
        except Exception as e:
            raise RuntimeError(e)
        plan = np.array([[p.pose.position.x, p.pose.position.y] for p in plan_msg.plan.poses])

        # calculate center of virtual circles that block left half of the hallway
        dist = np.cumsum( np.linalg.norm(plan[1:] - plan[:-1], axis=1) )
        idx_bgn, idx_end = np.searchsorted(dist, [1.5, dist[-1]-1.5]) # (1.0 + 0.5)^2 - 1.0^2 ~ 1.12 <1.2 + 0.3s buffer
        idx = slice(idx_bgn, idx_end, 4)    # install hallucination every 4 plan points.

        dx, dy = (plan[2:] - plan[:-2]).T
        theta = (np.arctan2(dy, dx) + np.pi/2.)[idx_bgn:idx_end:4]
        plan = plan[idx_bgn:idx_end:4]

        centers = plan + (RADIUS + 0.05) * np.array([np.cos(theta), np.sin(theta)]).T

        msg = PolygonStamped()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(9999.9)
        msg.polygon.points = [Point32(x, y, RADIUS) for x, y in centers]
        self.pub_hallucination.publish(msg)

        return super().active_cb()
    
    def done_cb(self, state, result):
        # clear hallucination
        self.clear_hallucination()
        return super().done_cb(state, result)

class D_PHHP(Baseline):
    def __init__(self, id, policy, network_dir):
        super().__init__(id)
        self.policy = policy
        self.network_dir = network_dir

        # storage
        self.raw_scan = torch.zeros(size=(640,), dtype=torch.float32)
        self.hal_scan = torch.zeros(size=(640,), dtype=torch.float32)

        # define subscriber
        self.sub_raw_scan = rospy.Subscriber(os.path.join(self.id, "scan_filtered"), LaserScan, self.raw_scan_cb)
        self.sub_hal_scan = rospy.Subscriber(os.path.join(self.id, "scan_hallucinated"), LaserScan, self.hal_scan_cb)

    def raw_scan_cb(self, msg):
        self.raw_scan = torch.nan_to_num(
            torch.tensor(msg.ranges, dtype=torch.float32) / (msg.range_max - msg.range_min), 
            nan=0.0
        )

    def hal_scan_cb(self, msg):
        self.raw_scan = torch.nan_to_num(
            torch.tensor(msg.ranges, dtype=torch.float32) / (msg.range_max - msg.range_min),
            nan=0.0
        )

    def reset(self, x: float, y: float, yaw: float, var: float = 0.01):
        # Load new policy from file
        self.policy.load_state_dict( torch.load(self.network_dir) )
        return super().reset(x, y, yaw, var)

    def active_cb(self):
        self.ttd = rospy.Time.now()
    
    def get_state(self):
        
        state = torch.zeros(size=(640*2+2+1,), dtype=torch.float32)

        # Store scan information
        state[0:640] = self.raw_scan
        state[640:1280] = self.hal_scan

        # Store velocity message
        try:
            vel_msg = rospy.wait_for_message(os.path.join(self.id, "cmd_vel"), Twist, timeout=0.1)
            state[1280] = vel_msg.linear.x
            state[1281] = vel_msg.angular.z
        except rospy.ROSException as e:
            pass

        # Store distance to goal
        x, y, yaw = self.trajectory[self.idx-1]
        gyaw = atan2( self.goal.target_pose.pose.position.y - y, self.goal.target_pose.pose.position.x - x )
        state[1282] = (gyaw - yaw + np.pi) / (2.*np.pi)

        return state

    def hallucinate(self, explore=False):
        state = self.get_state()
        if explore is True:
            action = np.random.uniform(low=-1.0, high=1.0, size=3)
        else:
            with torch.no_grad():
                action, _ = self.policy(state)
        reward = -1.0

        # convert (r, th, t) -> (global_x, global_y, t)
        r  = action[0]*4.0 + 5.0    #   1 ~  9
        th = action[1]*np.pi        # -pi ~ pi
        t  = action[2]*5.0 + 5.0    #   0 ~ 10

        x, y, yaw = self.trajectory[self.idx-1]
        global_x = x + r * cos(yaw + th)
        global_y = y + r * sin(yaw + th)

        msg = PolygonStamped()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(t)
        msg.polygon.points = [Point32(global_x, global_y, 1.0)]
        self.pub_hallucination.publish(msg)

        return state, action, reward

if __name__ == "__main__":
    # Only use for Debug!
    rospy.init_node("debug")
    rospy.sleep(1.0)

    marvin = Vanilla(id='marvin')

    marvin.goto(0, 0, 0, timeout=30)

    while not marvin.is_arrived():
        rospy.sleep(0.1)
    print(marvin.ttd)

    import matplotlib.pyplot as plt
    traj = marvin.trajectory[:marvin.idx]
    plt.scatter(traj.T[0], traj.T[1])
    plt.savefig("tt.png", dpi=300)
