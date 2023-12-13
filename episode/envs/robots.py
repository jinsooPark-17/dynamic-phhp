#/usr/bin/env python3
import os
import time
import torch
import numpy as np
from copy import deepcopy
from math import sin, cos, atan2

import rospy
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv import Empty
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from geometry_msgs.msg import PolygonStamped, Point32

def quaternion_to_yaw(q):
    # roll = atan2( 2.0*(q.w*q.x + q.y*q.z), 1.0 - 2.0*(q.x*q.x + q.y*q.y) )
    # pich = 2.0 * atan2(sqrt(1.0 + 2.0*(q.w*q.y - q.x*q.z)), sqrt(1.0 - 2.0*(q.w*q.y - q.x*q.z))) - PI/2.0
    # pich = arcsin( 2.0*(q.w*q.y - q.x*q.z) )
    yaw = atan2( 2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z) )
    return yaw

class AllinOne(object):
    def __init__(self, id: str = "", policy = None):
        # define variables
        self.id = id
        self.policy = policy

        self.ttd: float = None
        self.goal = MoveBaseGoal()
        self.pose = PoseWithCovarianceStamped().pose.pose   # Pose() msg
        self.trajectory = np.zeros(shape=(0,3), dtype=np.float32)
        self.traj_idx: int = 0
        self.__next_install = None
        self.__cycle = None

        # Define data storage
        self.raw_scan = torch.zeros(size=(640,), dtype=torch.float32)
        self.hal_scan = torch.zeros(size=(640,), dtype=torch.float32)
        self.cmd_vel  = torch.zeros(size=(  4,), dtype=torch.float32)

        # Connect to ROS MoveBase
        self.__move_base = SimpleActionClient(
            os.path.join(self.id, "move_base"),
            MoveBaseAction
        )
        self.connected = self.__move_base.wait_for_server(timeout=rospy.Duration(10.0))

        # Define ROS services
        self.__make_plan_srv = rospy.ServiceProxy(
            os.path.join(self.id, "move_base", "NavfnROS", "make_plan"),    # "$ID/move_base/NavfnROS/make_plan"
            GetPlan
        )
        self.__clear_costmap_srv = rospy.ServiceProxy(
            os.path.join(self.id, "move_base", "clear_costmaps"),
            Empty
        )
        self.__clear_hallucination_srv = rospy.ServiceProxy(
            os.path.join(self.id, "clear_virtual_circles"),
            Empty
        )

        # Define ROS publisher
        self.__pub_localize = rospy.Publisher(
            os.path.join(self.id, "initialpose"),
            PoseWithCovarianceStamped,
            queue_size=10
        )
        self.__pub_hallucination = rospy.Publisher(
            os.path.join(self.id, "add_circles"),
            PolygonStamped,
            queue_size=10
        )

        # Define ROS subscriber
        self.__sub_raw_scan = rospy.Subscriber(
            os.path.join(self.id, "scan_filtered"),
            LaserScan, 
            self.__raw_scan_cb
        )
        self.__sub_hal_scan = rospy.Subscriber(
            os.path.join(self.id, "scan_hallucinated"),
            LaserScan,
            self.__hal_scan_cb
        )
        self.__sub_cmd_vel = rospy.Subscriber(
            os.path.join(self.id, "cmd_vel"),
            Twist,
            self.__cmd_vel_cb
        )

    def __raw_scan_cb(self, msg):
        self.raw_scan = np.array(msg.ranges, np.float32) / (msg.range_max - msg.range_min)

    def __hal_scan_cb(self, msg):
        self.hal_scan = np.array(msg.ranges, np.float32) / (msg.range_max - msg.range_min)

    def __cmd_vel_cb(self, msg):
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.angular.z
        # store max value only
        self.cmd_vel[2] = max(self.cmd_vel[2], msg.linear.x)
        self.cmd_vel[3] = max(self.cmd_vel[3], msg.angular.z)

    def clear_costmap(self):
        rospy.wait_for_service( os.path.join(self.id, "move_base", "clear_costmaps") )
        try:
            self.__clear_costmap_srv()
        except rospy.ServiceException as e:
            raise RuntimeError(f"{self.id}: {e}")

    def clear_hallucination(self):
        rospy.wait_for_service( os.path.join(self.id, "clear_virtual_circles") )
        try:
            self.__clear_hallucination_srv()
        except rospy.ServiceException as e:
            raise RuntimeError(f"{self.id}: {e}")

    def make_plan(self, start, goal: MoveBaseGoal):
        rospy.wait_for_service( os.path.join(self.id, "move_base", "NavfnROS", "make_plan") )
        plan_req = GetPlanRequest()
        plan_req.start.header = goal.target_pose.header
        plan_req.start.pose = start
        plan_req.goal = goal.target_pose
        plan_req.tolerance = 0.1

        try:
            plan_msg = self.__make_plan_srv( plan_req )
            plan = np.array([[p.pose.position.x, p.pose.position.y] for p in plan_msg.plan.poses])
            return plan
        except rospy.ServiceException as e:
            raise RuntimeError(f"{self.id}: {e}")

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
        self.pose = msg.pose.pose

        self.__pub_localize.publish(msg)

    def dynamic_hallucinate(self, r, theta, t, radius=-0.5):
        # r, theta, t, radius in [-1.0, 1.0]
        radius = radius+1.0             # (0.0, 2.0)
        r      = r*4.0 + 4.5 + radius   # (0.5, 8.5) + radius
        theta  = theta*np.pi            # (-PI, PI)
        t      = t*5.0 + 5.0            # (0.0, 10.0)

        # convert relative (r,theta) to global (x,y) coordinate
        x, y, yaw = self.trajectory[self.traj_idx-1]

        cx = x + r*np.cos(yaw+theta)
        cy = y + r*np.sin(yaw+theta)

        # DO NOT!! install virtual obstacle if it covers goal point
        dx = self.goal.target_pose.pose.position.x - cx
        dy = self.goal.target_pose.pose.position.y - cy
        if dx*dx + dy*dy < (radius+0.5) * (radius+0.5):     # 0.5 is radius of the robot
            return

        msg = PolygonStamped()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(t)
        msg.polygon.points = [Point32(cx, cy, radius)]
        self.__pub_hallucination.publish(msg)

    def move(self, x: float, y: float, yaw: float, mode: str="vanilla", timeout: float=60.0, **kwargs):
        """
        x: x value of goal pose
        y: y value of goal pose
        yaw: yaw value of goal pose
        mode: drive mode of robot. [vanilla, baseline, custom, phhp, dynamic]
          - vanilla: Apply nothing
          - baseline: Apply maximun perceptual hallucination
          - custom: Apply custom perceptual hallucination
          - phhp: Apply minimum perceptual hallucination
          - dynamic: Apply dynamic hallucination
        timeout: set timeout for episode. (default: 60.0 seconds)
        """
        # Store trajectory from 20hz feedback loop
        self.trajectory = np.zeros(shape=(int(20*(timeout+1)), 3), dtype=np.float32)
        self.traj_idx = 0

        self.goal.target_pose.header.frame_id = os.path.join(self.id, "level_mux_map")
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.goal.target_pose.pose.position.x = x
        self.goal.target_pose.pose.position.y = y
        self.goal.target_pose.pose.orientation.z = sin(yaw/2.)
        self.goal.target_pose.pose.orientation.w = cos(yaw/2.)

        self.__movebase_args = dict()
        if mode == "vanilla":
            feedback_cb = self.vanilla_feedback_cb
        elif mode == "baseline":
            feedback_cb = self.static_hallucination_feedback_cb
            try:
                self.__movebase_args["active"] = False
                self.__movebase_args["detection_range"] = kwargs["detection_range"]
                self.__movebase_args["comms_topic"] = kwargs["comms_topic"]
                self.__movebase_args["radius"]  = 1.0
                self.__movebase_args["gap"]     = 0.05
                self.__movebase_args["p_begin"] = 0.0
                self.__movebase_args["p_end"]   = 1.0
            except KeyError as e:
                print("BASELINE mode require [detection_range, comms_topic] arguments!")
        elif mode == "custom":
            feedback_cb = self.static_hallucination_feedback_cb
            try:
                self.__movebase_args["active"] = False
                self.__movebase_args["detection_range"] = kwargs["detection_range"]
                self.__movebase_args["comms_topic"] = kwargs["comms_topic"]
                self.__movebase_args["radius"]  = kwargs["radius"]
                self.__movebase_args["gap"]     = kwargs["gap"]
                self.__movebase_args["p_begin"] = kwargs["p_begin"]
                self.__movebase_args["p_end"]   = kwargs["p_end"]
            except KeyError as e:
                print("CUSTOM mode require [detection_range, comms_topic, radius, gap, p_begin, p_end] arguments!")
        elif mode == "phhp":
            feedback_cb = self.static_hallucination_feedback_cb
            try:
                self.__movebase_args["active"] = False
                self.__movebase_args["detection_range"] = kwargs["detection_range"]
                self.__movebase_args["comms_topic"] = kwargs["comms_topic"]
                # Configuration for L-shape hallway in Table 1. (https://ieeexplore-ieee-org.ezproxy.lib.utexas.edu/document/10161327) 
                self.__movebase_args["radius"]  = 0.5122
                self.__movebase_args["gap"]     = 0.0539
                self.__movebase_args["p_begin"] = 0.4842
                self.__movebase_args["p_end"]   = 0.5000
            except KeyError as e:
                print("PHHP mode require [detection_range, comms_topic] arguments!")
        elif mode == "dynamic":
            self.policy = kwargs['policy']
            self.__cycle = kwargs['cycle']
            self.__next_install = rospy.Time.now() + rospy.Duration(self.__cycle)
            feedback_cb = self.dynamic_hallucination_feedback_cb

        self.ttd = rospy.Time.now()
        self.__move_base.send_goal(
            goal        = self.goal,
            feedback_cb = feedback_cb,
        )

    def vanilla_feedback_cb(self, feedback):
        # Timeout
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()

        # Store current pose of robot to trajectory log
        self.pose = feedback.base_position.pose
        self.trajectory[self.traj_idx] = [
            feedback.base_position.pose.position.x,
            feedback.base_position.pose.position.y,
            quaternion_to_yaw(feedback.base_position.pose.orientation)
        ]
        self.traj_idx = self.traj_idx + 1

    def static_hallucination_feedback_cb(self, feedback):
        # Timeout
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()

        # Store current pose of robot to trajectory log
        self.pose = feedback.base_position.pose
        yaw = quaternion_to_yaw(feedback.base_position.pose.orientation)
        self.trajectory[self.traj_idx] = [
            feedback.base_position.pose.position.x,
            feedback.base_position.pose.position.y,
            yaw
        ]
        self.traj_idx = self.traj_idx + 1

        if self.__movebase_args["active"] is True:
            # Return if static hallucination is already applied.
            return

        # Request current pose of opponent robot
        try:
            comms_msg = rospy.wait_for_message(topic=self.__movebase_args["comms_topic"], topic_type=PoseWithCovarianceStamped, timeout=0.10)
            comms_pose = np.array([comms_msg.pose.pose.position.x, comms_msg.pose.pose.position.y])
        except rospy.ROSException as e:
            return

        # if opponent robot is facing same direction, ignore
        comms_yaw = quaternion_to_yaw(comms_msg.pose.pose.orientation)
        if not (0.5*np.pi < np.abs(yaw - comms_yaw) <= 1.5*np.pi):
            return

        # Request plan of robot
        plan = self.make_plan( self.pose, self.goal )
        dist = np.cumsum( np.linalg.norm(plan[1:] - plan[:-1], axis=1) ) # monotonic increasing array

        # Check if opponent approaches within the detection range
        proximity = np.linalg.norm(plan[1:] - comms_pose, axis=1)
        min_idx = np.argmin(proximity)
        if proximity[min_idx] > 0.35:
            return

        if dist[min_idx] <= self.__movebase_args["detection_range"]:
            self.__movebase_args["active"] = True
            d_min = max(dist[min_idx]*self.__movebase_args["p_begin"], self.__movebase_args["radius"]+1.0)
            d_max = min(dist[min_idx]*self.__movebase_args["p_end"], dist[-1]-(self.__movebase_args["radius"]+1.0))
            if d_min > d_max:
                return

            idx_bgn, idx_end = np.searchsorted(dist, [d_min, d_max])

            dx, dy = (plan[2:] - plan[:-2]).T
            theta = np.arctan2(dy, dx)[idx_bgn:idx_end:4] + np.pi/2.
            centers = plan[idx_bgn+1:idx_end+1:4] + np.sign(self.__movebase_args["gap"])*(self.__movebase_args["radius"]+np.abs(self.__movebase_args["gap"])) * np.array([np.cos(theta), np.sin(theta)]).T

            msg = PolygonStamped()
            msg.header.stamp = rospy.Time.now() + rospy.Duration(9999.9)
            msg.polygon.points = [Point32(x, y, self.__movebase_args["radius"]) for x, y in centers]
            self.__pub_hallucination.publish(msg)

    def dynamic_hallucination_feedback_cb(self, feedback):
        # Timeout
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()

        # Store current pose of robot to trajectory log
        self.pose = feedback.base_position.pose
        self.trajectory[self.traj_idx] = [
            feedback.base_position.pose.position.x,
            feedback.base_position.pose.position.y,
            quaternion_to_yaw(feedback.base_position.pose.orientation)
        ]
        self.traj_idx = self.traj_idx + 1

        now = rospy.Time.now()
        if now > self.__next_install:
            self.__next_install = now + rospy.Duration(self.__cycle)
            s = self.get_state()
            with torch.no_grad():
                a, _ = self.policy(s, deterministic=True)
            self.dynamic_hallucinate(*a.squeeze())

    def stop(self):
        self.__move_base.cancel_all_goals()

    def is_running(self):
        if self.__move_base.get_result() is None:
            return True
        if type(self.ttd) is not float:
            self.ttd = (rospy.Time.now() - self.ttd).to_sec()
        return False

    def is_arrived(self):
        return (self.__move_base.get_state() == GoalStatus.SUCCEEDED)

    def get_state(self, ):
        state = torch.zeros(size=(640*2+2+1,), dtype=torch.float32)
        # Store scan information
        state[0:640]    = torch.from_numpy(np.nan_to_num(self.raw_scan, posinf=1.0))
        state[640:1280] = torch.from_numpy(np.nan_to_num(self.hal_scan, posinf=1.0))
        # Store velocity message
        state[1280] = self.cmd_vel[0]
        # state[1281] = self.cmd_vel[2] # Use max v value
        state[1281] = self.cmd_vel[1]
        # state[1281] = self.cmd_vel[3] # Use max w value
        self.cmd_vel.zero_()

        # Store angle to goal
        x, y, yaw = self.trajectory[self.traj_idx-1]
        gyaw = atan2( self.goal.target_pose.pose.position.y - y, self.goal.target_pose.pose.position.x - x )
        state[1282] = (gyaw - yaw + np.pi) / (2.*np.pi)

        return state.unsqueeze(0)

    def get_trajectory(self):
        return self.trajectory[:self.traj_idx]

if __name__ == "__main__":
    # Only use for Debug!
    rospy.init_node("debug")
    rospy.sleep(1.0)

    marvin = AllinOne(id='marvin')

    marvin.goto(0, 0, 0, timeout=30)

    while not marvin.is_arrived():
        rospy.sleep(0.1)
    print(marvin.ttd)

    import matplotlib.pyplot as plt
    traj = marvin.trajectory[:marvin.idx]
    plt.scatter(traj.T[0], traj.T[1])
    plt.savefig("tt.png", dpi=300)

