#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
import argparse
from tf_transformations import quaternion_from_euler, euler_from_quaternion  # Import for Euler/Quaternion conversion

class MoveItCommander(Node):
    def __init__(self, use_joint_positions):
        super().__init__("moveit_commander")
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        
        self.moveit2.max_velocity = 0.75
        self.moveit2.max_acceleration = 0.5
        
        self.use_joint_positions = use_joint_positions
        
        if self.use_joint_positions:
            self.get_logger().info("Joint commander initialized. Ready to accept joint configurations.")
        else:
            self.get_logger().info("Pose commander initialized. Ready to accept target positions.")

    def get_current_pose(self):
        """Get the current end effector pose and print it"""
        # Compute forward kinematics for the end effector
        fk_result = self.moveit2.compute_fk()
        
        if fk_result is None:
            self.get_logger().warn("Could not compute current pose")
            return None
            
        if isinstance(fk_result, list):
            current_pose = fk_result[0]  # Take first result if multiple
        else:
            current_pose = fk_result
            
        # Convert quaternion to Euler angles for more intuitive display
        quat = [
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        
        # Print current pose information
        self.get_logger().info("\nCurrent End Effector Pose:")
        self.get_logger().info(f"Position: x={current_pose.pose.position.x:.3f}, y={current_pose.pose.position.y:.3f}, z={current_pose.pose.position.z:.3f}")
        self.get_logger().info(f"Orientation (Quaternion): x={current_pose.pose.orientation.x:.3f}, y={current_pose.pose.orientation.y:.3f}, z={current_pose.pose.orientation.z:.3f}, w={current_pose.pose.orientation.w:.3f}")
        self.get_logger().info(f"Orientation (Euler RPY): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
        
        return current_pose

    def run(self):
        while rclpy.ok():
            try:
                if self.use_joint_positions:
                    # Joint position mode
                    input_str = input("\nEnter joint angles (radians, comma separated): ")
                    joint_positions = [float(x.strip()) for x in input_str.split(',')]
                    if len(joint_positions) != 6:
                        self.get_logger().error("Please enter exactly 6 joint values")
                        continue
                    self.moveit2.move_to_configuration(joint_positions)
                else:
                    # Cartesian position mode - show current pose first
                    current_pose = self.get_current_pose()
                    
                    input_str = input("\nEnter target position (x,y,z) and orientation (roll,pitch,yaw) as 6 comma-separated values: ")
                    values = [float(x.strip()) for x in input_str.split(',')]
                    if len(values) != 6:
                        self.get_logger().error("Please enter exactly 6 values (position x,y,z and orientation roll,pitch,yaw)")
                        continue
                    
                    position = Point(x=values[0], y=values[1], z=values[2])
                    roll, pitch, yaw = values[3], values[4], values[5]
                    
                    # Convert Euler angles to quaternion
                    quat = quaternion_from_euler(roll, pitch, yaw)
                    orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
                    
                    self.moveit2.move_to_pose(position=position, quat_xyzw=orientation)
                
                self.moveit2.wait_until_executed()
                
            except Exception as e:
                self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', action='store_true', help='Use joint position control')
    parser.add_argument('--pose', action='store_true', help='Use Cartesian pose control')
    args = parser.parse_args()
    
    # Default to joint control if neither flag is set
    use_joint_positions = args.joint or not args.pose
    
    commander = MoveItCommander(use_joint_positions)
    commander.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()