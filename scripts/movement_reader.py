#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
import argparse
from tf_transformations import quaternion_from_euler, euler_from_quaternion  # Import for Euler/Quaternion conversion
import os
from ament_index_python.packages import get_package_share_directory

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

    def get_current_pose(self):
        """Get the current end effector pose and print it"""
        fk_result = self.moveit2.compute_fk()
        
        if fk_result is None:
            self.get_logger().warn("Could not compute current pose")
            return None
            
        current_pose = fk_result[0] if isinstance(fk_result, list) else fk_result
        
        quat = [
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        
        self.get_logger().info("\nCurrent End Effector Pose:")
        self.get_logger().info(f"Position: x={current_pose.pose.position.x:.3f}, y={current_pose.pose.position.y:.3f}, z={current_pose.pose.position.z:.3f}")
        self.get_logger().info(f"Orientation (RPY): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
        
        return current_pose

    def execute_command(self, command_str):
        try:
            if self.use_joint_positions:
                joint_positions = [float(x.strip()) for x in command_str.split(',')]
                if len(joint_positions) != 6:
                    self.get_logger().error("Invalid command: Need exactly 6 joint values")
                    return False
                
                self.get_logger().info(f"Executing joint command: {joint_positions}")
                self.moveit2.move_to_configuration(joint_positions)
            else:
                values = [float(x.strip()) for x in command_str.split(',')]
                if len(values) != 6:
                    self.get_logger().error("Invalid command: Need exactly 6 values (position x,y,z and RPY)")
                    return False
                
                position = Point(x=values[0], y=values[1], z=values[2])
                roll, pitch, yaw = values[3], values[4], values[5]
                quat = quaternion_from_euler(roll, pitch, yaw)
                orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
                
                self.get_logger().info(f"Executing pose command: Position={values[0:3]}, RPY={values[3:6]}")
                self.moveit2.move_to_pose(position=position, quat_xyzw=orientation)
            
            self.moveit2.wait_until_executed()
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error executing command: {e}")
            return False

    def run_from_file(self, filename):
        try:
            # Get the package share directory (installed directory)
            package_share_dir = get_package_share_directory('ar4_practice')
            
            # Construct the full path to the file in the scripts folder (installed directory)
            file_path = os.path.join(package_share_dir, '..', 'scripts', filename)
            
            # If the file is not found in the installed directory, check the source directory
            if not os.path.isfile(file_path):
                self.get_logger().warn(f"File not found in installed directory: {file_path}")
                
                # Check the source directory
                source_dir = os.path.join(os.getcwd(), 'src', 'ar4_practice', 'scripts', filename)
                self.get_logger().info(f"Looking for command file in source directory: {source_dir}")
                
                if os.path.isfile(source_dir):
                    file_path = source_dir
                else:
                    self.get_logger().error(f"File not found in source directory: {source_dir}")
                    return False
            
            self.get_logger().info(f"Reading commands from file: {file_path}")
            
            # Read and execute commands from the file
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    self.get_current_pose()
                    self.get_logger().info(f"\nExecuting command from line {line_num}: {line}")
                    
                    if not self.execute_command(line):
                        self.get_logger().error(f"Skipping invalid command at line {line_num}")
                        continue
            
            self.get_logger().info("Finished executing all commands from file")
            return True
        
        except Exception as e:
            self.get_logger().error(f"Error in run_from_file: {e}")
            return False         

    def run_interactive(self):
        mode = "joint positions" if self.use_joint_positions else "Cartesian poses"
        self.get_logger().info(f"Running in interactive {mode} mode. Enter commands below:")
        
        while rclpy.ok():
            self.get_current_pose()
            
            if self.use_joint_positions:
                prompt = "\nEnter joint angles (6 values, comma separated): "
            else:
                prompt = "\nEnter position and RPY (6 values, comma separated x,y,z,roll,pitch,yaw): "
            
            command_str = input(prompt)
            self.execute_command(command_str)

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', action='store_true', help='Use joint position control')
    parser.add_argument('--pose', action='store_true', help='Use Cartesian pose control')
    parser.add_argument('--file', type=str, help='File containing commands (one per line)')
    args = parser.parse_args()
    
    use_joint_positions = args.joint or not args.pose
    commander = MoveItCommander(use_joint_positions)
    
    if args.file:
        commander.run_from_file(args.file)
    else:
        commander.run_interactive()
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()