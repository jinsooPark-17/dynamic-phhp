#/usr/bin/env python3
import os
import time
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

def quaternion_to_yaw(q):
    # roll = atan2( 2.0*(q.w*q.x + q.y*q.z), 1.0 - 2.0*(q.x*q.x + q.y*q.y) )
    # pich = 2.0 * atan2(sqrt(1.0 + 2.0*(q.w*q.y - q.x*q.z)), sqrt(1.0 - 2.0*(q.w*q.y - q.x*q.z))) - PI/2.0
    # pich = arcsin( 2.0*(q.w*q.y - q.x*q.z) )
    yaw = atan2( 2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z) )
    return yaw

class AllinOne(object):
    def __init__(self, id: str = "", policy = None, debug: str=""):
        # define variables
        start_time = time.time()
        self.id = id
        self.policy = policy

        self.ttd: float = None
        self.goal = MoveBaseGoal()
        self.pose = PoseWithCovarianceStamped().pose.pose   # Pose() msg
        self.trajectory = np.zeros(shape=(0,3), dtype=np.float32)
        self.traj_idx: int = 0

        # Define data storage
        self.raw_scan = torch.zeros(size=(640,), dtype=torch.float32)
        self.hal_scan = torch.zeros(size=(640,), dtype=torch.float32)
        self.cmd_vel  = torch.zeros(size=(  4,), dtype=torch.float32)

        # Connect to ROS MoveBase
        start_time = time.time()
        self.__move_base = SimpleActionClient(
            os.path.join(self.id, "move_base"),
            MoveBaseAction
        )
        if debug: print(f"  {debug}: Define move_base took {time.time() - start_time:.3f} sec", flush=True)
        start_time = time.time()
        self.connected = self.__move_base.wait_for_server(timeout=rospy.Duration(10.0))
        if debug: print(f"  {debug}: move_base.wait_for_server took {time.time() - start_time:.3f} sec", flush=True)

        # Define ROS services
        start_time = time.time()
        self.__make_plan_srv = rospy.ServiceProxy(
            os.path.join(self.id, "move_base", "make_plan"),    # "$ID/move_base/NavfnROS/make_plan"
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
        if debug: print(f"  {debug}: Define ros service took {time.time() - start_time:.3f} sec", flush=True)

        # Define ROS publisher
        start_time = time.time()
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
        if debug: print(f"  {debug}: Define ROS publisher took {time.time() - start_time:.3f} sec", flush=True)

        # Define ROS subscriber
        start_time = time.time()
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
        if debug: print(f"  {debug}: Define subscriber took {time.time() - start_time:.3f} sec", flush=True)

    def __raw_scan_cb(self, msg):
        self.raw_scan = torch.nan_to_num(
            torch.tensor(msg.ranges, dtype=torch.float32) / (msg.range_max - msg.range_min),
            nan = 0.0
        )

    def __hal_scan_cb(self, msg):
        self.hal_scan = torch.nan_to_num(
            torch.tensor(msg.ranges, dtype=torch.float32) / (msg.range_max - msg.range_min),
            nan = 0.0
        )

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

    def make_plan(self, req):
        rospy.wait_for_service( os.path.join(self.id, "move_base", "NavfnROS", "make_plan") )
        # rospy.wait_for_service( os.path.join(self.id, "move_base", "make_plan") )
        try:
            plan_msg = self.__make_plan_srv( req )
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

    def hallucinate(self):
        raise NotImplementedError()

    def perceptual_hallucination(self, radius=1.0, dr=0.05, p_min=0.2, p_max=0.8):
        # Get plan as numpy array
        plan_req = GetPlanRequest()
        plan_req.start = plan_req.goal = self.goal.target_pose
        plan_req.start.pose = self.pose
        plan_req.tolerance = 0.1
        plan = self.make_plan(plan_req)

        # calculate location of virtual circles that block left half of plan
        dist = np.cumsum( np.linalg.norm(plan[1:] - plan[:-1], axis=1) ) # monotonic increasing array
        d_min = max(dist[-1]*p_min, 1.5)
        d_max = min(dist[-1]*p_max, dist[-1]-1.5)
        idx_bgn, idx_end = np.searchsorted(dist, [d_min, d_max])

        dx, dy = (plan[2:] - plan[:-2]).T
        theta = np.arctan2(dy, dx)[idx_bgn:idx_end:4] + np.pi/2.
        centers = plan[idx_bgn+1:idx_end+1:4] + (radius + dr) * np.array([np.cos(theta), np.sin(theta)]).T

        msg = PolygonStamped()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(9999.9)
        msg.polygon.points = [Point32(x, y, radius) for x, y in centers]
        self.__pub_hallucination.publish(msg)

    def move(self, x: float, y: float, yaw: float, mode: str="vanilla", timeout: float=60.0, **kwargs):
        # Store trajectory from 20hz feedback loop
        self.trajectory = np.zeros(shape=(int(20*(timeout+1)), 3), dtype=np.float32)
        self.traj_idx = 0

        self.goal.target_pose.header.frame_id = os.path.join(self.id, "level_mux_map")
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.goal.target_pose.pose.position.x = x
        self.goal.target_pose.pose.position.y = y
        self.goal.target_pose.pose.orientation.z = sin(yaw/2.)
        self.goal.target_pose.pose.orientation.w = cos(yaw/2.)

        if mode == "vanilla":
            pass
        elif mode == "baseline":
            self.perceptual_hallucination(radius=0.5, p_min=0.2, p_max=0.8)
        elif mode == "d-phhp":
            self.policy.load_state_dict( torch.load(kwargs['network_dir']) )# load policy

        self.ttd = rospy.Time.now()
        self.__move_base.send_goal(
            goal        = self.goal,
            # active_cb   = None,
            feedback_cb = self.feedback_cb,
            # done_cb     = None
        )

    def feedback_cb(self, feedback):
        self.pose = feedback.base_position.pose
        self.trajectory[self.traj_idx] = [
            feedback.base_position.pose.position.x,
            feedback.base_position.pose.position.y,
            quaternion_to_yaw(feedback.base_position.pose.orientation)
        ]
        self.traj_idx = self.traj_idx + 1

        # Timeout
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()

    def stop(self):
        self.__move_base.cancel_all_goals()

    def is_running(self):
        if self.__move_base.get_result() is None:
            return True
        if type(self.ttd) is not float:
            self.ttd = (rospy.Time.now() - self.ttd).to_sec()
        return False

    def is_arrived(self):
        return (self.__move_base.get_state() == 3)

    def get_state(self, ):
        state = torch.zeros(size=(640*2+2+1,), dtype=torch.float32)
        # Store scan information
        state[0:640]    = self.raw_scan
        state[640:1280] = self.hal_scan
        # Store velocity message
        state[1280] = self.cmd_vel[0]
        state[1281] = self.cmd_vel[1]
        # state[1281] = self.cmd_vel[3]
        self.cmd_vel.zero_()

        # Store angle to goal
        x, y, yaw = self.trajectory[self.traj_idx-1]
        gyaw = atan2( self.goal.target_pose.pose.position.y - y, self.goal.target_pose.pose.position.x - x )
        state[1282] = (gyaw - yaw + np.pi) / (2.*np.pi)

        return state

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

