
import rclpy
from rclpy.node import Node
from moveit_action_client import MoveItActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf_transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix, euler_from_matrix
import numpy as np
import time
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState


from ar4_robot_py import AR4Robot



def main():
    """Example usage of the AR4Robot class with movement completion detection"""
    rclpy.init()
    robot = AR4Robot()    

    #robot.move_to_home()
    target_position_preferred = [0.536455, 0.0, 0.313363]  # x, y, z in preferred frame
    target_orientation_preferred = [0, 0.0, 0.0]  # roll, pitch, yaw in preferred frame
    robot.move_to_pose_preferred_frame(
        target_position_preferred, 
        target_orientation_preferred, 
        frame_id="base_link"
    )
    time.sleep(0.5)
    target_position_preferred = [0.506455, -0.02, 0.273363]  # x, y, z in preferred frame
    target_orientation_preferred = [0, 0.0, 0.0]  # roll, pitch, yaw in preferred frame
    robot.move_to_pose_preferred_frame(
        target_position_preferred, 
        target_orientation_preferred, 
        frame_id="base_link"
    )
    
    #robot.move_to_home()
    
    rclpy.spin(robot.moveit_client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()