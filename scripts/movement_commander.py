#!/usr/bin/env python3

import rclpy
import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion
from ar4_robot import AR4_ROBOT


def main(args=None):
    rclpy.init(args=args)
    #frame = "base_link"
    frame = "end_effector_link"
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', action='store_true', help='Use joint position control')
    parser.add_argument('--pose', action='store_true', help='Use Cartesian pose control')
    args = parser.parse_args()
    
    # Default to joint control if neither flag is set
    use_joint_positions = args.joint or args.pose
    
    robot = AR4_ROBOT(use_joint_positions)
    
    try:
        while rclpy.ok():
            if use_joint_positions:
                # Joint position mode
                while True:
                    input_str = input("\nEnter joint angles (radians, comma separated): ")
                    try:
                        joint_positions = [float(x.strip()) for x in input_str.split(',')]
                        if len(joint_positions) != 6:
                            raise ValueError("Please enter exactly 6 joint values")
                        robot.move_to_joint_positions(joint_positions)
                        break  # Exit the loop on valid input
                    except ValueError as e:
                        robot.get_logger().error(f"Invalid input: {e}")
                        continue  # Prompt the user again
            else:
                # Cartesian position mode
                #current_pose = robot.get_current_pose("base_link")
                current_pose = robot.get_current_pose(frame)
                if current_pose is not None:
                    position, orientation = current_pose
                    robot.get_logger().info(f"Current Pose: Position={position}, Orientation={orientation}")
                
                while True:
                    input_str = input("\nEnter target position (x,y,z) and orientation (roll,pitch,yaw) as 6 comma-separated values: ")
                    try:
                        values = [float(x.strip()) for x in input_str.split(',')]
                        if len(values) != 6:
                            raise ValueError("Please enter exactly 6 values (position x,y,z and orientation roll,pitch,yaw)")
                        
                        position = [values[0], values[1], values[2]]
                        euler_angles = [values[3], values[4], values[5]]
                        
                        
                        robot.move_to_pose(position, euler_angles,frame)
                        break  # Exit the loop on valid input
                    except ValueError as e:
                        robot.get_logger().error(f"Invalid input: {e}")
                        robot.get_logger().info(f"Current Pose: Position={position}, Orientation={orientation}")
                        continue  # Prompt the user again
    except Exception as e:
        robot.get_logger().error(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()