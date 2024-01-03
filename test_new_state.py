import os
import numpy as np
from copy import deepcopy
from math import sin, cos, atan2
from skimage import feature

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

def quaternion_to_yaw(q):
    # roll = atan2( 2.0*(q.w*q.x + q.y*q.z), 1.0 - 2.0*(q.x*q.x + q.y*q.y) )
    # pitch = 2.0 * atan2(sqrt(1.0 + 2.0*(q.w*q.y - q.x*q.z)), sqrt(1.0 - 2.0*(q.w*q.y - q.x*q.z))) - PI/2.0
    # pitch = arcsin( 2.0*(q.w*q.y - q.x*q.z) )
    yaw = atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))
    return yaw

class Costmap:
    def __init__(self, costmap_msg, sigma=0.5):
        self.w = costmap_msg.info.width
        self.h = costmap_msg.info.height
        self.resolution = costmap_msg.info.resolution
        self.origin = np.array([costmap_msg.info.origin.position.x, costmap_msg.info.origin.position.y])
        self.costmap = np.array(costmap_msg.data, dtype=np.float64).reshape(self.w, self.h) / 100.

class Agent:
    def __init__(self, id):
        self.__id = id
        # pre-define scan related variables
        scan_msg = rospy.wait_for_message(os.path.join(id, 'scan_filtered'), LaserScan)
        self.odom = None
        self.theta = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        self.range_max = scan_msg.range_max
        self.range_min = scan_msg.range_min
        self.raw_scan = None
        self.hal_scan = None

        self.__odom_cb = rospy.Subscriber(os.path.join(id, 'odom'), Odometry, self.__odom_callback)
        self.__raw_scan_cb = rospy.Subscriber(os.path.join(id, 'scan_filtered'), LaserScan, self.__raw_scan_callback)
        self.__hal_scan_cb = rospy.Subscriber(os.path.join(id, 'scan_hallucinated'), LaserScan, self.__hal_scan_callback)

        # Store wall data from global costmap
        self.map = Costmap(rospy.wait_for_message(os.path.join(id, 'move_base', 'global_costmap', 'costmap'), OccupancyGrid))

    # Define callbacks
    def __odom_callback(self, odom_msg):
        self.odom = odom_msg.pose.pose  # position, orientation
    def __raw_scan_callback(self, scan_msg):
        self.raw_scan = scan_msg.ranges
    def __hal_scan_callback(self, scan_msg):
        self.hal_scan = scan_msg.ranges

    def scan_to_pointcloud(self, scan):
        # Change to use TF when deploy with real BWIbots.
        # Launch python3 node that accept state message and return action.
        # 1. find sensor origin: odom -> sensor (0.15875m to x-axis)
        x, y = self.odom.position.x, self.odom.position.y
        yaw = quaternion_to_yaw(self.odom.orientation)

        dx = 0.15875    # displacement of sensor
        sensor_x, sensor_y = x+dx*cos(yaw), y+dx*sin(yaw)
        scan_x = sensor_x + scan * np.cos(self.theta+yaw)
        scan_y = sensor_y + scan * np.sin(self.theta+yaw)

        # Coordinate to index
        valid_idx = np.isfinite(scan_x + scan_y)
        x_idx = ((scan_x - self.map.origin[0]) / self.map.resolution).astype(np.int)
        y_idx = ((scan_y - self.map.origin[1]) / self.map.resolution).astype(np.int)
        uncharted = (self.map.costmap[x_idx[valid_idx], y_idx[valid_idx]] > 0.99)
        state = np.zeros_like(scan)
        # state[valid_idx][uncharted] = (scan[valid_idx][uncharted]-self.range_min) /(self.range_max-self.range_min) 
        state[valid_idx][uncharted] = scan[valid_idx][uncharted]

        return state, scan

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rospy.init_node('dev_state', anonymous=True)
    
    rospy.sleep(0.5)
    marvin = Agent('marvin')
    rospy.sleep(0.5)

    state, scan = marvin.scan_to_pointcloud( marvin.raw_scan )
    plt.plot(scan, c='k')
    plt.plot(state, c='r')
    plt.savefig('new_info.png')