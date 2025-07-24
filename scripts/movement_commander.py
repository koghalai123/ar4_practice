#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
import argparse
from tf_transformations import quaternion_from_euler, euler_from_quaternion  # Import for Euler/Quaternion conversion
import numpy as np
from tf_transformations import euler_matrix, euler_from_matrix

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

        # Reference transformation for the end effector
        self.reference_translation = [0.0, 0.0, 0.0]  # Example translation (x, y, z)
        self.reference_rotation = [0.0, 0.0, np.pi *1/2]  # Example rotation (roll, pitch, yaw)
        self.transformation_matrix = create_transformation_matrix(self.reference_translation, self.reference_rotation)
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)

        self.angle_offsets = {
            "roll": 0,  # Example offset in radians
            "pitch": 1.571,
            "yaw": 1.571
        }

    def get_current_pose(self):
        """Get the current end effector pose and print it."""
        fk_result = self.moveit2.compute_fk()
        
        if fk_result is None:
            self.get_logger().warn("Could not compute current pose")
            return None
            
        if isinstance(fk_result, list):
            current_pose = fk_result[0]  # Take first result if multiple
        else:
            current_pose = fk_result

        # Convert quaternion to Euler angles
        quat = [
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)

        # Apply transformation to the pose
        position = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z, 1.0])
        transformed_position = np.dot(self.transformation_matrix, position)[:3]
        transformed_orientation = euler_from_matrix(
            np.dot(self.transformation_matrix[:3, :3], euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3]),
            axes='sxyz'  # Example alternative axes order
        )
        globalOrientation = [transformed_orientation[1] + self.angle_offsets["roll"],
                             transformed_orientation[0] + self.angle_offsets["pitch"], 
                             transformed_orientation[2] + self.angle_offsets["yaw"]]

        # Print transformed pose
        self.get_logger().info("\nTransformed End Effector Pose:")
        self.get_logger().info(f"Position: x={transformed_position[0]:.3f}, y={transformed_position[1]:.3f}, z={transformed_position[2]:.3f}")
        self.get_logger().info(f"Orientation (Euler RPY): roll={globalOrientation[0]:.3f}, pitch={globalOrientation[1]:.3f}, yaw={globalOrientation[2]:.3f}")
        
        return transformed_position, transformed_orientation

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
                    
                    position = np.array([values[0], values[1], values[2], 1.0])
                    transformed_position = np.dot(self.inverse_transformation_matrix, position)[:3]

                    
                    roll = values[4] - self.angle_offsets["roll"]
                    pitch = values[3] - self.angle_offsets["pitch"]
                    yaw = values[5] - self.angle_offsets["yaw"]
                    
                    transformed_orientation_matrix = np.dot(self.inverse_transformation_matrix[:3, :3], euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3])
                    transformed_orientation = euler_from_matrix(transformed_orientation_matrix)

                    # Convert transformed orientation to quaternion
                    quat = quaternion_from_euler(transformed_orientation[0], transformed_orientation[1], transformed_orientation[2])
                    orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

                    self.moveit2.move_to_pose(position=Point(x=transformed_position[0], y=transformed_position[1], z=transformed_position[2]), quat_xyzw=orientation)
                
                self.moveit2.wait_until_executed()
                
            except Exception as e:
                self.get_logger().error(f"Error: {e}")

def create_transformation_matrix(translation, euler_angles):
    """Create a 4x4 homogeneous transformation matrix."""
    transform = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')
    transform[:3, 3] = translation
    return transform

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', action='store_true', help='Use joint position control')
    parser.add_argument('--pose', action='store_true', help='Use Cartesian pose control')
    args = parser.parse_args()
    
    # Default to joint control if neither flag is set
    #use_joint_positions = args.joint or not args.pose
    use_joint_positions = args.pose or args.joint
    
    commander = MoveItCommander(use_joint_positions)
    commander.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()