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
    def __init__(self, id, map_frame):
        self.id = id
        self.ttd = None
        self.goal = None
        self.__map_frame = os.path.join(id, 'level_mux_map')

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

    def clear_costmap(self):
        rospy.wait_for_service( os.path.join(self.id, "move_base", "clear_costmaps") )
        try:
            self.__clear_costmap_srv()
        except rospy.ServiceException as e:
            raise RuntimeError("{}: {}".format(self.id, e))

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

    def move(self, x, y, yaw, timeout=60.0):
        self.ttd = rospy.Time.now()

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
        # change timeout value
        self.goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(timeout)
        self.__move_base.send_goal(
            self.goal, 
            active_cb=self.active_cb, 
            feedback_cb=self.feedback_cb, 
            done_cb=self.done_cb
        )

    def wait_for_result(self):
        self.__move_base.wait_for_result()

    def active_cb(self):
        print(f"Active callback initiated: {(rospy.Time.now()-self.ttd).to_sec()}")

    def feedback_cb(self, feedback):
        if feedback.base_position.header.stamp > self.goal.target_pose.header.stamp:
            self.stop()

        print(f"Feedback callback initiated: {(rospy.Time.now()-self.ttd).to_sec()}")
        print(f"Current location: ({feedback.base_position.pose.position.x:.2f},{feedback.base_position.pose.position.y:.2f})")
        print(f"\tget_result: {self.__move_base.get_result()}")
        print(f"\tget_state: {self.__move_base.get_state()}")
        print(f"\tget_goal_status_text: {self.__move_base.get_goal_status_text()}")

    def done_cb(self, state, result):
        print(f"Done callback initiated: {(rospy.Time.now()-self.ttd).to_sec()}")
        print(f"state: {state}")
        print(f"result: {result}")

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


if __name__ == '__main__':
    rospy.init_node('dev_agent', anonymous=True)
    
    marvin = Movebase('marvin')
    rospy.sleep(0.5)

    print("Test move function")
    marvin.move(-5.0, 0., np.pi, timeout=5.0)
    marvin.wait_for_result()

    print("Test resume function")
    marvin.resume()
    marvin.wait_for_result()

    print("Test move_and_wait function")
    marvin.move_and_wait(0.0, 0.0, 0.0)