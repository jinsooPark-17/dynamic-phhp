import os
import numpy as np
from math import sin, cos, atan2
from scipy.spatial.distance import directed_hausdorff

import rospy
from actionlib import SimpleActionClient
from std_srvs.srv import Empty
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped

def quaternion_to_yaw(q):
    yaw = atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))
    return yaw

class Movebase:
    def __init__(self, id, map_frame='level_mux_map'):
        self.id = id
        self.ttd = None
        self.goal = None
        self.pose = None
        self.__map_frame = os.path.join(id, map_frame)

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
        msg.header.frame_id = self.__map_frame
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
        self.pose = msg.pose.pose
    def clear_costmap(self):
        rospy.wait_for_service( os.path.join(self.id, "move_base", "clear_costmaps") )
        try:
            self.__clear_costmap_srv()
        except rospy.ServiceException as e:
            raise RuntimeError("{}: {}".format(self.id, e))

    def move(self, x, y, yaw, timeout=60.0):
        self.pose = None
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.goal.target_pose.header.frame_id = self.__map_frame
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
        self.pose = None
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
        self.pose = feedback.base_position.pose
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

import matplotlib.pyplot as plt
class Agent(Movebase):
    def __init__(self, id, map_frame='level_mux_map'):
        super().__init__(id, map_frame)

        # Define scan related variables
        sample_scan_msg = rospy.wait_for_message(
            os.path.join(id, 'scan_filtered'), 
            LaserScan
        )
        self.theta = np.linspace(sample_scan_msg.angle_min, sample_scan_msg.angle_max, len(sample_scan_msg.ranges))[::-1]
        self.raw_scan = None

        self.__sub_raw_scan = rospy.Subscriber(
            os.path.join(id, 'scan_filtered'),
            LaserScan,
            self.__raw_scan_cb
        )
        plt.figure()
        plt.xlim(-25.0, 5.0)
        plt.ylim( -1.0, 1.0)

    def __raw_scan_cb(self, scan_msg):
        sx, sy = self.scan_to_global_pointcloud(scan_msg.ranges)
        plt.scatter(sx, sy, c='k', s=1)

    def scan_to_global_pointcloud(self, scan):
        dx = 0.15875
        yaw = quaternion_to_yaw(self.pose.orientation)
        sensor_x = self.pose.position.x + dx*cos(yaw)
        sensor_y = self.pose.position.y + dx*sin(yaw)

        scan_x = sensor_x + scan * np.cos(self.theta+yaw)
        scan_y = sensor_y + scan * np.sin(self.theta+yaw)
        finite_idx = np.isfinite(scan_x) * np.isfinite(scan_y)

        return scan_x[finite_idx], scan_y[finite_idx]

if __name__ == '__main__':
    rospy.init_node('dev_agent', anonymous=True)
    rospy.sleep(1.0)
    
    marvin = Movebase('marvin')

    print("Test move function")
    marvin.move(-5.0, 0., np.pi, timeout=60.0)
    while not marvin.is_arrived():
        rospy.sleep(0.01)
    plt.savefig('scan_history.png')