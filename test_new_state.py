import os
import numpy as np
from copy import deepcopy
from skimage import feature

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

def quaternion_to_yaw(q):
    # roll = atan2( 2.0*(q.w*q.x + q.y*q.z), 1.0 - 2.0*(q.x*q.x + q.y*q.y) )
    # pitch = 2.0 * atan2(sqrt(1.0 + 2.0*(q.w*q.y - q.x*q.z)), sqrt(1.0 - 2.0*(q.w*q.y - q.x*q.z))) - PI/2.0
    # pitch = arcsin( 2.0*(q.w*q.y - q.x*q.z) )
    yaw = np.arctan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))
    return yaw

class Costmap:
    def __init__(self, costmap_msg, sigma=0.5):
        self.w = costmap_msg.info.width
        self.h = costmap_msg.info.height
        self.resolution = costmap_msg.info.resolution
        self.origin = np.array([costmap_msg.info.origin.position.x, costmap_msg.info.origin.position.y])
        self.costmap = np.array(costmap_msg.data, dtype=np.float64).reshape(self.w, self.h) / 100.

        # Extract wall information from costmap
        self.edge = feature.canny(self.costmap, sigma=sigma)
        self.wall_xy = np.argwhere(self.edge == 1.0)
        self.wall_index = self.wall_xy.dot([1, self.w])



class Agent:
    def __init__(self, id):
        self.__id = id
        self.amcl = None
        self.odom = None
        self.odom_amcl = None

        self.theta = None
        self.raw_scan = None
        self.hal_scan = None

        self.__amcl_cb = rospy.Subscriber(os.path.join(id, 'amcl_pose'), PoseWithCovarianceStamped, self.__amcl_callback)
        self.__odom_cb = rospy.Subscriber(os.path.join(id, 'odom'), Odometry, self.__odom_callback)
        self.__raw_scan_cb = rospy.Subscriber(os.path.join(id, 'scan_filtered'), LaserScan, self.__raw_scan_callback)
        self.__hal_scan_cb = rospy.Subscriber(os.path.join(id, 'scan_hallucinated'), LaserScan, self.__hal_scan_callback)

        # Store wall data from global costmap
        self.map = Costmap(rospy.wait_for_message(os.path.join(id, 'move_base', 'global_costmap', 'costmap'), OccupancyGrid))

    # Define callbacks
    def __amcl_callback(self, amcl_msg):
        self.amcl = amcl_msg.pose.pose          # position, orientation
        self.odom_amcl = deepcopy(self.odom)    # odom position at given AMCL message
    def __odom_callback(self, odom_msg):
        self.odom = odom_msg.pose.pose  # position, orientation
    def __raw_scan_callback(self, scan_msg):
        if self.theta is None:
            self.theta = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        self.raw_scan = scan_msg.ranges
    def __hal_scan_callback(self, scan_msg):
        self.hal_scan = scan_msg.ranges

    def scan_to_pointcloud(self, scan):
        # 1. find sensor origin: amcl -> odom -> sensor (0.15875m to x-axis)
        
        return

"""
range_min
range_max
"""

odom_msg = rospy.wait_for_message('/marvin/odom', Odometry)
odom = odom_msg.pose.pose.position
yaw = quat2yaw(odom_msg.pose.pose.orientation)
