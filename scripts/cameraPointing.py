#!/usr/bin/env python3

import rclpy
import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion
from ar4_robot import AR4_ROBOT
import numpy as np

def main(args=None):
    rclpy.init(args=args)
    frame = "end_effector_link"
    use_joint_positions = 0
    robot = AR4_ROBOT(use_joint_positions)
    for i in range(10):
        try:
            random_numbers = np.random.rand(3) 
            random_pos = (random_numbers-0.5)*0.1
            xOffset = 0
            yOffset = 0
            zOffset = robot.pos_offsets["z"]
            x = xOffset + random_pos[0]
            y = yOffset + random_pos[1]
            z = zOffset + random_pos[2]
            
            pitch = -np.arctan2(x, z)-np.pi/2
            r = x**2 + z**2
            roll = -np.arctan2(y, r)
            yaw = 0
            
            current_pose = robot.get_current_pose(frame)
            position, orientation = current_pose
            robot.get_logger().info(f"Current Pose: Position={position}, Orientation={orientation}")

            position = [random_pos[0], random_pos[1], random_pos[2]]
            euler_angles = [roll,pitch,yaw]
            robot.move_to_pose(position, euler_angles,frame)
        except Exception as e:
            robot.get_logger().error(f"Error: {e}")



if __name__ == "__main__":
    main()