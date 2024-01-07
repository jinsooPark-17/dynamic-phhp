import os
import numpy as np
from math import sin, cos, atan2
from scipy.spatial.distance import directed_hausdorff

import rospy
from actionlib import SimpleActionClient
from std_srvs.srv import Empty
from nav_msgs.srv import GetPlan, GetPlanRequest
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped

def quaternion_to_yaw(q):
    yaw = atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))
    return yaw

class Costmap:
    # Data class
    def __init__(self, costmap_msg, sigma=0.5):
        self.w = costmap_msg.info.width
        self.h = costmap_msg.info.height
        self.resolution = costmap_msg.info.resolution
        self.origin = np.array([costmap_msg.info.origin.position.x, costmap_msg.info.origin.position.y])
        self.costmap = np.array(costmap_msg.data, dtype=np.float64).reshape(self.w, self.h) / 100.

class Movebase(object):
    def __init__(self, id, map_frame='level_mux_map'):
        self.id = id
        self.ttd = None
        self.goal = None
        self.map_frame = os.path.join(id, map_frame)

        # Define move_base parameters
        self.__move_base = SimpleActionClient(os.path.join(self.id, 'move_base'), MoveBaseAction)
        if not self.__move_base.wait_for_server(timeout=rospy.Duration(10.0)):
            raise TimeoutError("{}: move_base does not respond.".format(self.id))

        # Define services
        self.__clear_costmap_srv = rospy.ServiceProxy(os.path.join(self.id, 'move_base', 'clear_costmaps'), Empty)

        # Define publisher
        self.__pub_localize = rospy.Publisher(os.path.join(self.id, 'initialpose'), PoseWithCovarianceStamped, queue_size=10)

        # Define subscriber
        # self.__sub_odom = rospy.

    def localize(self, x, y, yaw, var=0.01):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
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
        for _ in range(10):
            self.__pub_localize.publish(msg)
            rospy.sleep(0.01)

    def clear_costmap(self):
        rospy.wait_for_service( os.path.join(self.id, "move_base", "clear_costmaps") )
        try:
            self.__clear_costmap_srv()
        except rospy.ServiceException as e:
            raise RuntimeError("{}: {}".format(self.id, e))

    def move(self, x, y, yaw, timeout=60.0):
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.goal.target_pose.header.frame_id = self.map_frame
        self.goal.target_pose.pose.position.x = x
        self.goal.target_pose.pose.position.y = y
        self.goal.target_pose.pose.orientation.z = sin(yaw/2.)
        self.goal.target_pose.pose.orientation.w = cos(yaw/2.)

        self.__move_base.send_goal(
            self.goal, 
            active_cb=self.active_cb, 
            feedback_cb=self.feedback_cb, 
            done_cb=self.done_cb
        )
    def move_and_wait(self, x, y, yaw):
        self.move(x, y, yaw, timeout=9999.9)
        self.__move_base.wait_for_result()
    def resume(self, timeout=30.0):
        if self.goal is None:
            print("{}: ERROR! no previous goal found.".format(self.id))
            return

        # Change timeout value
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.__move_base.send_goal(
            self.goal, 
            active_cb=self.active_cb, 
            feedback_cb=self.feedback_cb, 
            done_cb=self.done_cb
        )
    def wait_for_result(self):
        self.__move_base.wait_for_result()
    def stop(self):
        self.__move_base.cancel_all_goals()

    def active_cb(self):
        self.ttd = rospy.Time.now()
    def feedback_cb(self, feedback):
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()
    def done_cb(self, state, result):
        self.ttd = (rospy.Time.now() - self.ttd).to_sec()

    def is_running(self):
        if self.__move_base.get_result() is None:
            return True
        if type(self.ttd) is not float:
            self.ttd = (rospy.Time.now() - self.ttd).to_sec()
        return False

    def is_arrived(self):
        return (self.__move_base.get_state() == GoalStatus.SUCCEEDED)

class Agent(Movebase):
    def __init__(
            self, id, 
            num_scan_history, 
            sensor_horizon, 
            plan_interval, 
            map_frame='level_mux_map'):
        super(Agent, self).__init__(id, map_frame)

        # Define internal parameters
        self.pose = None
        self.map = Costmap(rospy.wait_for_message(os.path.join(id, 'move_base', 'global_costmap', 'costmap'), OccupancyGrid))

        # Define state related variables
        sample_scan_msg = rospy.wait_for_message(os.path.join(id, 'scan_filtered'), LaserScan)
        n_scan = len(sample_scan_msg.ranges)
        self.theta = np.linspace(sample_scan_msg.angle_min, sample_scan_msg.angle_max, n_scan)[::-1]
        self.num_scan_history = num_scan_history
        self.max_range = sensor_horizon
        self.raw_scan, self.raw_scan_idx = np.zeros((num_scan_history, n_scan)), 0
        self.hal_scan, self.hal_scan_idx = np.zeros((num_scan_history, n_scan)), 0

        self.cmd_vel = dict(stamp=rospy.Time(0), data=np.zeros(4))  # v, w, max(v), max(w)

        self.prev_plan = None   # previous plan to calculate directed_hausdorff distance
        self.curr_plan = None   # current plan to present as state
        self.sensor_horizon = sensor_horizon
        self.plan_interval = np.arange(0.0, self.sensor_horizon+plan_interval, plan_interval)

        # Define subscirbers
        self.__sub_odom = rospy.Subscriber(os.path.join(id, 'odom'), Odometry, self.__odom_cb)
        self.__sub_plan = rospy.Subscriber(os.path.join(id, 'move_base', 'NavfnROS', 'plan'), Path, self.__plan_cb)
        self.__sub_cmd_vel = rospy.Subscriber(os.path.join(id, 'cmd_vel'), Twist, self.__cmd_vel_cb)
        self.__sub_raw_scan = rospy.Subscriber(os.path.join(id, 'scan_filtered'), LaserScan, self.__raw_scan_cb)
        self.__sub_hal_scan = rospy.Subscriber(os.path.join(id, 'scan_hallucinated'), LaserScan, self.__hal_scan_cb)

        # Define services
        self.__make_plan_srv = rospy.ServiceProxy(os.path.join(self.id, "move_base", "NavfnROS", "make_plan"), GetPlan)

        while not rospy.is_shutdown():
            if self.pose is not None:
                break
            rospy.sleep(1e-3)

    def __odom_cb(self, odom_msg):
        self.pose = odom_msg.pose.pose
        # TODO: Convert odom to map frame during real deployment
    
    def __plan_cb(self, plan_msg):
        self.curr_plan = np.array([[p.pose.position.x, p.pose.position.y] for p in plan_msg.poses])

    def __cmd_vel_cb(self, vel_msg):
        self.cmd_vel['stamp'] = rospy.Time.now()    # Stamp to exclude pre-dated message
        self.cmd_vel['data'][0] = vel_msg.linear.x
        self.cmd_vel['data'][1] = vel_msg.angular.z
        self.cmd_vel['data'][2] = self.cmd_vel['data'][[0,2]].max()
        self.cmd_vel['data'][3] = self.cmd_vel['data'][[1,3]].max()

    def exclude_known_information(self, scan):
        # Convert scan to global pointcloud
        dx = 0.15875
        yaw = quaternion_to_yaw(self.pose.orientation)
        sensor_x = self.pose.position.x + dx*cos(yaw)
        sensor_y = self.pose.position.y + dx*sin(yaw)

        scan_x = sensor_x + scan * np.cos(self.theta + yaw)
        scan_y = sensor_y + scan * np.sin(self.theta + yaw)
        valid_idx = np.logical_and(np.isfinite(scan_x), np.isfinite(scan_y))

        scan_x_pixel = ((scan_x[valid_idx] - self.map.origin[0]) / self.map.resolution).round().astype(int)
        scan_y_pixel = ((scan_y[valid_idx] - self.map.origin[1]) / self.map.resolution).round().astype(int)
        valid_idx[valid_idx] = (self.map.costmap[scan_x_pixel, scan_y_pixel] < 1.0)

        uncharted_scan = np.zeros_like(scan)
        uncharted_scan[valid_idx] = scan[valid_idx]
        return uncharted_scan
    
    def __raw_scan_cb(self, scan_msg):
        # Only store uncharted information
        self.raw_scan[self.raw_scan_idx] = scan_msg.ranges
        # self.raw_scan[self.raw_scan_idx] = self.exclude_known_information(scan_msg.ranges)
        self.raw_scan_idx = (self.raw_scan_idx + 1) % self.num_scan_history

    def __hal_scan_cb(self, scan_msg):
        # Only store uncharted information
        self.hal_scan[self.hal_scan_idx] = self.exclude_known_information(scan_msg.ranges)
        self.hal_scan_idx = (self.hal_scan_idx + 1) % self.num_scan_history

    def make_plan(self, goal_x, goal_y, goal_yaw):
        service_req = GetPlanRequest()
        service_req.start.header = self.map_frame
        service_req.start.pose = self.pose
        service_req.goal.pose.position.x = goal_x
        service_req.goal.pose.position.y = goal_y
        service_req.goal.pose.orientation.z = sin(goal_yaw/2.)
        service_req.goal.pose.orientation.w = cos(goal_yaw/2.)
        service_req.tolerance = 0.1

        rospy.wait_for_message(os.path.join(self.id, 'move_base', 'NavfnROS', 'make_plan'))
        try:
            plan_msg = self.__make_plan_srv(service_req)
            plan = np.array([[p.pose.position.x, p.pose.position.y] for p in plan_msg.plan.poses])
        except rospy.ServiceException as e:
            raise RuntimeError("{}: make_plan failed\n{}".format(self.id, e))

        return plan

    def move(self, x, y, yaw, timeout=60.0):
        # Make global plan to the goal
        self.curr_plan = self.prev_plan = self.make_plan(x, y, yaw)        
        super(Agent, self).move(x, y, yaw, timeout)

    def find_valid_plan(self, plan):
        p = np.array([self.pose.position.x, self.pose.position.y])
        dist_to_robot = np.linalg.norm(plan-p, axis=1)
        valid_idx = np.argmin(dist_to_robot)
        return plan[valid_idx:]

    def filter_plan(self, valid_plan):
        valid_plan = np.insert(valid_plan, 0, [self.pose.position.x, self.pose.position.y], axis=0)
        d = np.cumsum(np.linalg.norm(valid_plan[1:] - valid_plan[:-1], axis=1))
        idx = np.searchsorted(d, self.plan_interval)
        filtered_plan = valid_plan[idx]
        return filtered_plan

    def get_state(self, cmd_vel_type='max'):
        # State #1: scan
        raw_scans = np.roll(self.raw_scan, -self.raw_scan_idx) / self.sensor_horizon
        hal_scans = np.roll(self.hal_scan, -self.hal_scan_idx) / self.sensor_horizon
        ## leave only visible scans
        raw_scans[raw_scans > 1.0] = 0.
        hal_scans[hal_scans > 1.0] = 0.

        # State #2: plan
        prev_plan = self.find_valid_plan(self.prev_plan)
        curr_plan = self.find_valid_plan(self.curr_plan)
        hausdorff = -directed_hausdorff(prev_plan, curr_plan)
        self.prev_plan = self.curr_plan.copy()

        plan_x, plan_y = self.filter_plan(curr_plan).T
        ego_plan = np.vstack((np.hypot(plan_x,plan_y)/self.sensor_horizon, np.arctan2(plan_y,plan_x)/np.pi)).T

        # State #3: cmd_vel
        if cmd_vel_type == 'max':
            vw = self.cmd_vel['data'][2:].copy()
        elif (rospy.Time.now() - self.cmd_vel['stamp']).to_sec() < 0.1:
            vw = self.cmd_vel['data'][:2].copy()
        else:
            vw = np.zeros(2)

        state = dict(
            scan=np.vstack((raw_scans, hal_scans)), 
            plan=ego_plan, 
            hausdorff_dist=hausdorff, 
            vw=vw
        )
        return state

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    os.makedirs('debug')
    rospy.init_node('dev_agent', anonymous=True)
    rospy.sleep(1.0)
    
    marvin = Agent(id='marvin', num_scan_history=1, sensor_horizon=8.0, plan_interval=0.5)
    marvin.move(-5.0, 0., np.pi, timeout=60.0)

    frame_id = 0
    while not marvin.is_arrived():
        state = marvin.get_state()

        # save ego-centric frame
        plt.figure(figsize=(8,8)); plt.xlim(-8.5, 8.5); plt.ylim(-8.5, 8.5)
        scan_x = state['scan'] * np.cos(marvin.theta) * marvin.sensor_horizon
        scan_y = state['scan'] * np.sin(marvin.theta) * marvin.sensor_horizon
        plt.scatter(scan_x[0], scan_y[0], c='r', s=1)
        plt.scatter(scan_x[1], scan_y[1], c='b', s=1)

        r, theta = state['plan'][:,0] * marvin.sensor_horizon, state['plan'][:,1] * np.pi
        plan_x = r * np.cos(theta)
        plan_y = r * np.sin(theta)
        plt.plot(plan_x, plan_y, 'k:')

        plt.title(f"{state['vw']}")
        plt.savefig(f"debug/{frame_id:05d}.png")
        plt.close()
        frame_id += 1
        rospy.sleep(0.5)
