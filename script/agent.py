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
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped, PolygonStamped, Point32

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
        self.move_base = SimpleActionClient(os.path.join(self.id, 'move_base'), MoveBaseAction)
        if not self.move_base.wait_for_server(timeout=rospy.Duration(10.0)):
            raise TimeoutError("{}: move_base does not respond.".format(self.id))

        # Define services
        self.__clear_costmap_srv = rospy.ServiceProxy(os.path.join(self.id, 'move_base', 'clear_costmaps'), Empty)

        # Define publisher
        self.__pub_localize = rospy.Publisher(os.path.join(self.id, 'initialpose'), PoseWithCovarianceStamped, queue_size=10)

        # Define subscriber

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

        self.move_base.send_goal(
            self.goal, 
            active_cb=self.active_cb, 
            feedback_cb=self.feedback_cb, 
            done_cb=self.done_cb
        )
    def move_and_wait(self, x, y, yaw):
        self.move(x, y, yaw, timeout=9999.9)
        self.move_base.wait_for_result()
    def resume(self, timeout=30.0):
        if self.goal is None:
            print("{}: ERROR! no previous goal found.".format(self.id))
            return

        # Change timeout value
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.move_base.send_goal(
            self.goal, 
            active_cb=self.active_cb, 
            feedback_cb=self.feedback_cb, 
            done_cb=self.done_cb
        )
    def wait_for_result(self):
        self.move_base.wait_for_result()
    def stop(self):
        self.move_base.cancel_all_goals()

    def active_cb(self):
        self.ttd = rospy.Time.now()
    def feedback_cb(self, feedback):
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()
    def done_cb(self, state, result):
        self.ttd = (rospy.Time.now() - self.ttd).to_sec()

    def is_running(self):
        if self.move_base.get_result() is None:
            return True
        if type(self.ttd) is not float:
            self.ttd = (rospy.Time.now() - self.ttd).to_sec()
        return False

    def is_arrived(self):
        return (self.move_base.get_state() == GoalStatus.SUCCEEDED)

class Agent(Movebase):
    def __init__(
            self, id, 
            num_scan_history, 
            sensor_horizon, 
            plan_interval, 
            map_frame='level_mux_map',
            radius=0.35):
        super(Agent, self).__init__(id, map_frame)

        # Define internal parameters
        self.radiis = radius
        self.pose = None
        self.map = Costmap(rospy.wait_for_message(os.path.join(id, 'move_base', 'global_costmap', 'costmap'), OccupancyGrid))

        # Define state related variables
        sample_scan_msg = rospy.wait_for_message(os.path.join(id, 'scan_filtered'), LaserScan)
        n_scan = len(sample_scan_msg.ranges)
        self.theta = np.linspace(sample_scan_msg.angle_min, sample_scan_msg.angle_max, n_scan)[::-1]
        self.num_scan_history = num_scan_history
        self.raw_scan, self.raw_scan_idx = np.zeros((num_scan_history, n_scan)), 0
        self.hal_scan, self.hal_scan_idx = np.zeros((num_scan_history, n_scan)), 0

        self.cmd_vel = np.zeros(2)  # max(v), max(w)

        self.prev_plan = None   # previous plan to calculate directed_hausdorff distance
        self.curr_plan = None   # current plan to present as state
        self.sensor_horizon = sensor_horizon
        self.plan_interval = np.arange(0.0, self.sensor_horizon+plan_interval, plan_interval)[1:]

        # Define services
        self.__make_plan_srv = rospy.ServiceProxy(os.path.join(self.id, "move_base", "NavfnROS", "make_plan"), GetPlan)
        self.__clear_hallucination_srv = rospy.ServiceProxy(os.path.join(self.id, "clear_virtual_circles"), Empty)

        # Define publishers
        self.__pub_hallucination = rospy.Publisher(os.path.join(self.id, "add_circles"), PolygonStamped, queue_size=10)

        # Define subscirbers
        self.__sub_odom = rospy.Subscriber(os.path.join(id, 'odom'), Odometry, self.__odom_cb)
        self.__sub_plan = rospy.Subscriber(os.path.join(id, 'move_base', 'NavfnROS', 'plan'), Path, self.__plan_cb)
        self.__sub_cmd_vel = rospy.Subscriber(os.path.join(id, 'cmd_vel'), Twist, self.__cmd_vel_cb)
        self.__sub_raw_scan = rospy.Subscriber(os.path.join(id, 'scan_filtered'), LaserScan, self.__raw_scan_cb)
        self.__sub_hal_scan = rospy.Subscriber(os.path.join(id, 'scan_hallucinated'), LaserScan, self.__hal_scan_cb)

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
        self.cmd_vel[0] = max(self.cmd_vel[0], vel_msg.linear.x)
        self.cmd_vel[1] = max(self.cmd_vel[1], vel_msg.angular.z)

    def exclude_known_information(self, scan):
        if not isinstance(scan, np.ndarray):
            scan = np.array(scan)
        # Convert scan to global pointcloud
        dx = 0.15875
        yaw = quaternion_to_yaw(self.pose.orientation)
        sensor_x = self.pose.position.x + dx*cos(yaw)
        sensor_y = self.pose.position.y + dx*sin(yaw)

        scan_x = sensor_x + scan * np.cos(self.theta + yaw)
        scan_y = sensor_y + scan * np.sin(self.theta + yaw)
        valid_idx = np.logical_and(np.isfinite(scan_x), np.isfinite(scan_y))

        scan_x_pixel = ((scan_x - self.map.origin[0]) / self.map.resolution).round().astype(int)
        scan_y_pixel = ((scan_y - self.map.origin[1]) / self.map.resolution).round().astype(int)
        try:
            valid_idx[valid_idx] = (self.map.costmap[scan_x_pixel[valid_idx], scan_y_pixel[valid_idx]] < 0.95)
        except IndexError:
            return np.zeros_like(scan)

        uncharted_scan = np.zeros_like(scan)
        uncharted_scan[valid_idx] = scan[valid_idx]

        return uncharted_scan
    
    def __raw_scan_cb(self, scan_msg):
        # Only store uncharted information
        self.raw_scan[self.raw_scan_idx] = self.exclude_known_information(scan_msg.ranges)
        self.raw_scan_idx = (self.raw_scan_idx + 1) % self.num_scan_history

    def __hal_scan_cb(self, scan_msg):
        # Only store uncharted information
        self.hal_scan[self.hal_scan_idx] = self.exclude_known_information(scan_msg.ranges)
        self.hal_scan_idx = (self.hal_scan_idx + 1) % self.num_scan_history

    def make_plan(self, goal_x, goal_y, goal_yaw):
        service_req = GetPlanRequest()
        service_req.start.header.frame_id = service_req.goal.header.frame_id = self.map_frame
        service_req.start.pose = self.pose
        service_req.goal.pose.position.x = goal_x
        service_req.goal.pose.position.y = goal_y
        service_req.goal.pose.orientation.z = sin(goal_yaw/2.)
        service_req.goal.pose.orientation.w = cos(goal_yaw/2.)
        service_req.tolerance = 0.1

        rospy.wait_for_service(os.path.join(self.id, 'move_base', 'NavfnROS', 'make_plan'))
        try:
            plan_msg = self.__make_plan_srv(service_req)
        except rospy.ServiceException as e:
            raise RuntimeError("{}: make_plan failed\n{}".format(self.id, e))

        return np.array([[p.pose.position.x, p.pose.position.y] for p in plan_msg.plan.poses])

    def clear_hallucination(self):
        rospy.wait_for_service(os.path.join(self.id, "clear_virtual_circles"))
        try:
            self.__clear_hallucination_srv()
        except rospy.ServiceException as e:
            raise RuntimeError("{}: clear hallucination failed\n{}".format(self.id, e))

    def move(self, x, y, yaw, timeout=60.0, mode='vanilla', **kwargs):
        if mode=='baseline' and not kwargs.keys() >= {'traffic'}:
            raise KeyError("Baseline mode require traffic argument!:\n\t[left, right]")
        if mode=='phhp' and not kwargs.keys() >= {'comms_topic', 'traffic'}:
            raise KeyError("PHHP mode require comms_topic argument!:\n\tamcl_pose topic of opponent robot")

        # Make global plan to the goal
        self.curr_plan = self.prev_plan = self.make_plan(x, y, yaw)

        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.goal.target_pose.header.frame_id = self.map_frame
        self.goal.target_pose.pose.position.x = x
        self.goal.target_pose.pose.position.y = y
        self.goal.target_pose.pose.orientation.z = sin(yaw/2.)
        self.goal.target_pose.pose.orientation.w = cos(yaw/2.)

        self.__kwargs = kwargs
        active_cb = (self.baseline_active_cb if mode == 'baseline' else self.active_cb)
        feedback_cb = (self.phhp_feedback_cb if mode == 'phhp' else self.feedback_cb)
        self.move_base.send_goal(
            self.goal, 
            active_cb=active_cb, 
            feedback_cb=feedback_cb, 
            done_cb=self.done_cb
        )

    def baseline_active_cb(self):
        self.ttd = rospy.Time.now()
        
        # Define baseline parameters
        gap      = 0.05
        radius   = 0.2
        traffic  = self.__kwargs['traffic']

        # Place virtual objects all around the global plan
        #   no plan
        plan = self.curr_plan.copy()
        dx, dy = (plan[2:] - plan[:-2]).T
        theta = np.arctan2(dy, dx) + (np.pi/2. if traffic=='left' else -np.pi/2.)

        msg = PolygonStamped()
        msg.header.stamp = self.goal.target_pose.header.stamp
        for (x,y), th in zip(plan[80:-80:4], theta[79:-81:4]):
            cx = x + (radius+gap)*cos(th)
            cy = y + (radius+gap)*sin(th)
            msg.polygon.points.append(Point32(cx, cy, radius))
        self.__pub_hallucination.publish(msg)

    def phhp_feedback_cb(self, feedback):
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()

        # if PHHP already placed virtual object, ignore
        if self.__kwargs.keys() >= {'installed'}:
            return

        # Get other robot's position
        try:
            msg = rospy.wait_for_message(self.__kwargs['comms_topic'], PoseWithCovarianceStamped, timeout=0.1)
            opponent = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        except rospy.ROSException as e:
            return
        # if my plan does not overlap with opponent robot, ignore
        plan = self.find_valid_plan(self.curr_plan)
        dist_to_opponent = np.linalg.norm(plan-opponent, axis=1)
        if dist_to_opponent.min() > 0.3:
            return
        # if distance to opponent is less than 8m, activate PHHP
        """
        PHHP parameters:
            radius = 0.5122
            dr = 0.0539
            p_bgn = 0.4842 (3.8736m with detection range: 8.0m)
            p_end = 0.5001 (4.0008m with detection range: 8.0m)
        """
        opponent_idx = np.argmin(dist_to_opponent)
        dist = np.linalg.norm(plan[1:]-plan[:-1], axis=1).cumsum()
        if dist[opponent_idx] < 8.0:
            dx, dy = (plan[2:] - plan[:-2]).T
            theta = np.arctan2(dy, dx) + (np.pi/2. if self.__kwargs['traffic']=='left' else -np.pi/2.)
            idx_bgn, idx_end = np.searchsorted(dist, [3.8736, 4.0008])

            msg = PolygonStamped()
            msg.header.stamp = self.goal.target_pose.header.stamp
            for (x, y), th in zip(plan[idx_bgn:idx_end], theta[idx_bgn-1:idx_end-1]):
                cx = x + 0.5661 * cos(th)
                cy = y + 0.5661 * sin(th)
                msg.polygon.points.append(Point32(cx, cy, 0.5122))  # (x,y,r)
            self.__pub_hallucination.publish(msg)
            self.__kwargs['installed'] = True

    def done_cb(self, state, result):
        self.ttd = (rospy.Time.now() - self.ttd).to_sec()
        self.clear_hallucination()

    def find_valid_plan(self, plan):
        if plan.size == 0:
            return np.array([self.pose.position.x, self.pose.position.y])
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

    def get_state(self):
        # State #1: scan
        raw_scans = np.roll(self.raw_scan, -self.raw_scan_idx) / self.sensor_horizon
        hal_scans = np.roll(self.hal_scan, -self.hal_scan_idx) / self.sensor_horizon
        ## leave only visible scans
        raw_scans[raw_scans > 1.0] = 0.
        hal_scans[hal_scans > 1.0] = 0.

        # State #2: plan
        curr_plan = self.find_valid_plan(self.curr_plan)
        if self.prev_plan.size == 0:
            hausdorff = 0.5
        else:
            prev_plan = self.find_valid_plan(self.prev_plan)
            hausdorff, _, _ = directed_hausdorff(prev_plan, curr_plan)
        self.prev_plan = self.curr_plan.copy()

        if self.curr_plan.size == 0:
            ego_plan = np.zeros((self.plan_interval.shape[0], 2))
        else:
            x = self.pose.position.x
            y = self.pose.position.y
            yaw = quaternion_to_yaw(self.pose.orientation)
            plan_x, plan_y = (self.filter_plan(curr_plan) - [x, y]).T
            ego_plan = np.vstack((np.hypot(plan_y, plan_x), np.arctan2(plan_y, plan_x) - yaw )).T
            ego_plan = ego_plan / [self.sensor_horizon, np.pi]

        # State #3: cmd_vel
        vw = self.cmd_vel.copy()
        self.cmd_vel[:] = 0.

        state = dict(
            scan=np.vstack((raw_scans, hal_scans)), 
            plan=ego_plan, 
            hausdorff_dist=hausdorff, 
            vw=vw
        )
        return state
    
    def action(self, valid, x, y, t, r=0.2):
        # SOMETHING WIERD! RE-WRITE!
        """
            valid: If True, install virtual obstacle with given (x, y, t, r) value.             [-1., 1.]
            x: proportion of target plan along the ego-centric plan divided by sensor horizon.  [-1., 1.]
            y: perpendicular distance from center of virtual circle to the target plan.         [-1., 1.]
            t: proportion of lifetime of virtual circle divided by max lifetime(default: 10.0). [-1., 1.]
        """
        if valid < -0.:
            return

        # convert action to values
        x = (x+1.)/2. * self.sensor_horizon # 0. ~ sensor_horizon (default: 0 ~ 8m)
        y = 5 * r * y                       # -5r ~ 5r (default: -1.0 ~ 1.0 m)
        t = (t+1.)*5.                       # 0.0 ~ 10.0 seconds

        # Find target plan from [x] value
        curr_plan = self.find_valid_plan(self.curr_plan)
        dist_to_robot = np.linalg.norm(curr_plan[1:] - curr_plan[:-1]).cumsum()
        target_plan_idx = np.searchsorted(dist_to_robot, x) + 1

        # Calculate center of virtual obstacle (vo)
        plan_x, plan_y = curr_plan[target_plan_idx]
        dx, dy = curr_plan[target_plan_idx+1] - curr_plan[target_plan_idx-1]
        plan_prep_theta = atan2(dy, dx) - np.pi/2.

        vo_x = plan_x + y * cos(plan_prep_theta)
        vo_y = plan_y + y * sin(plan_prep_theta)

        # filter invalid case
        if np.hypot(vo_x - self.pose.position.x, vo_y - self.pose.position.y) < self.radius + r + 0.3:
            return
        if np.hypot(vo_x - self.goal.target_pose.pose.position.x, vo_y - self.goal.target_pose.pose.position.y) < self.radius + r + 0.3:
            return

        # Create virtual obstacle!
        msg = PolygonStamped()
        msg.header.stamp = rospy.Time.now() + rospy.Duration(t)
        msg.polygon.points = [Point32(vo_x, vo_y, r)]
        self.__pub_hallucination.publish(msg)

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    if not os.path.exists('debug'):
        os.makedirs('debug')
    rospy.init_node('dev_agent', anonymous=True)
    rospy.sleep(1.0)
    
    marvin = Agent(id='marvin', num_scan_history=1, sensor_horizon=8.0, plan_interval=0.5)
    marvin.move(-5.0, 0., np.pi, timeout=60.0)

    frame_id = 0
    while not marvin.is_arrived():
        start_time = time.time()
        state = marvin.get_state()
        print(time.time() - start_time, state)

        # save ego-centric frame
        plt.figure(figsize=(8,8)); plt.xlim(-8.5, 8.5); plt.ylim(-8.5, 8.5)
        scan_x = state['scan'] * np.cos(marvin.theta) * marvin.sensor_horizon
        scan_y = state['scan'] * np.sin(marvin.theta) * marvin.sensor_horizon
        plt.scatter(scan_x[0], scan_y[0], c='r', s=2)

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
