#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_action_client import MoveItActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf_transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix, euler_from_matrix
import numpy as np
import time

def create_transformation_matrix(translation, euler_angles):
    """Create a 4x4 homogeneous transformation matrix."""
    transform = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')
    transform[:3, 3] = translation
    return transform

class AR4Robot:
    def __init__(self):
        """Initialize the AR4 Robot interface"""
        rclpy.init()
        self.moveit_client = MoveItActionClient()
        
        # Default scaling factors for safety
        self.default_velocity_scaling = 0.3
        self.default_acceleration_scaling = 0.3
        
        # Reference transformation for the end effector
        self.reference_translation = [0.0, 0.0, 0.0]
        self.reference_rotation = [0.0, 0.0, np.pi * 1 / 2]
        self.transformation_matrix = create_transformation_matrix(self.reference_translation, self.reference_rotation)
        self.inverse_transformation_matrix = np.linalg.inv(self.transformation_matrix)

        self.angle_offsets = {
            "roll": 0.0,
            "pitch": 1.571,
            "yaw": 1.571
        }
        
        self.pos_offsets = {
            "x": 0.32783003,
            "y": -0.00699888,
            "z": 0.47477099
        }
        
    def from_preferred_frame(self, position, euler_angles, old_reference_frame="base_link", new_reference_frame="base_link"):
        """Convert from preferred reference frame to MoveIt internal frame"""
        if new_reference_frame == old_reference_frame:
            position = np.array([position[0], position[1], position[2], 1.0])
        elif new_reference_frame == "end_effector_link" and old_reference_frame == "base_link":
            position = np.array([position[0], position[1], position[2], 1.0]) - np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"], 0.0])        
        elif new_reference_frame == "base_link" and old_reference_frame == "end_effector_link":
            position = np.array([position[0], position[1], position[2], 1.0]) + np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"], 0.0])        

        transformed_position = (np.dot(self.inverse_transformation_matrix, position)[:3])
        
        roll = euler_angles[1] - self.angle_offsets["pitch"]
        pitch = euler_angles[0] - self.angle_offsets["roll"]
        yaw = euler_angles[2] - self.angle_offsets["yaw"]
        
        transformed_orientation_matrix = np.dot(self.inverse_transformation_matrix[:3, :3], euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3])
        transformed_orientation = euler_from_matrix(transformed_orientation_matrix, axes='sxyz')

        return transformed_position, transformed_orientation

    def to_preferred_frame(self, position, euler_angles, reference_frame="base_link"):
        """Convert from MoveIt internal frame to preferred reference frame"""
        # Apply transformation matrix
        position_homogeneous = np.array([position[0], position[1], position[2], 1.0])
        transformed_position = np.dot(self.transformation_matrix, position_homogeneous)[:3]
        
        # Convert orientation
        orientation_matrix = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')[:3, :3]
        transformed_orientation_matrix = np.dot(self.transformation_matrix[:3, :3], orientation_matrix)
        transformed_euler = euler_from_matrix(transformed_orientation_matrix, axes='sxyz')
        
        # Apply angle offsets
        roll = transformed_euler[1] + self.angle_offsets["pitch"]
        pitch = transformed_euler[0] + self.angle_offsets["roll"]
        yaw = transformed_euler[2] + self.angle_offsets["yaw"]
        
        transformed_orientation = np.array([pitch, roll, yaw])
        
        # Apply position offsets based on reference frame
        if reference_frame == "end_effector_link":
            transformed_position = transformed_position - np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"]])
        elif reference_frame == "base_link":
            transformed_position = transformed_position + np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"]])
            
        return transformed_position, transformed_orientation
    
    def get_current_joint_state(self):
        """Get the current joint state as a dictionary"""
        return self.moveit_client.get_current_joint_values()
    
    def get_current_pose_preferred_frame(self, reference_frame="base_link", link_name="link_6"):
        """Get current end effector pose in preferred reference frame"""
        # Get pose from MoveIt
        pose_stamped = self.moveit_client.get_end_effector_pose(link_name)
        if pose_stamped is None:
            return None, None
            
        # Extract position and orientation
        pos = pose_stamped.pose.position
        orient = pose_stamped.pose.orientation
        position = np.array([pos.x, pos.y, pos.z])
        
        # Convert quaternion to euler
        quat = [orient.x, orient.y, orient.z, orient.w]
        euler_angles = np.array(euler_from_quaternion(quat))
        
        # Convert to preferred frame
        preferred_position, preferred_orientation = self.to_preferred_frame(position, euler_angles, reference_frame)
        
        return preferred_position, preferred_orientation
    
    def print_current_state(self, prefix="Current state", reference_frame="base_link"):
        """Print the current joint state and pose in preferred reference frame"""
        # Print joint state
        joints = self.get_current_joint_state()
        if joints:
            self.moveit_client.get_logger().info(f"{prefix} - Joint positions:")
            for joint, value in joints.items():
                self.moveit_client.get_logger().info(f"  {joint}: {value:.3f} rad ({np.degrees(value):.1f}°)")
        else:
            self.moveit_client.get_logger().warn("No joint state available")
            
        # Print end effector pose in preferred frame
        position, orientation = self.get_current_pose_preferred_frame(reference_frame)
        if position is not None and orientation is not None:
            self.moveit_client.get_logger().info(f"{prefix} - End effector pose in preferred frame ({reference_frame}):")
            self.moveit_client.get_logger().info(f"  Position: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f} m")
            self.moveit_client.get_logger().info(f"  Orientation (RPY): roll={orientation[0]:.3f} ({np.degrees(orientation[0]):.1f}°), "
                                 f"pitch={orientation[1]:.3f} ({np.degrees(orientation[1]):.1f}°), "
                                 f"yaw={orientation[2]:.3f} ({np.degrees(orientation[2]):.1f}°)")
        else:
            self.moveit_client.get_logger().warn("Could not get end effector pose in preferred frame")
    
    def wait_for_movement_complete(self, timeout=10.0):
        """
        Wait for robot to stop moving by monitoring joint velocities
        :param timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        self.moveit_client.get_logger().info("Waiting for movement to complete...")
        
        while time.time() - start_time < timeout:
            # Check if we have current joint state with velocities
            if self.moveit_client._current_joint_state is not None:
                joint_state = self.moveit_client._current_joint_state
                
                # Check if velocities are available and near zero
                if hasattr(joint_state, 'velocity') and len(joint_state.velocity) > 0:
                    max_velocity = max(abs(v) for v in joint_state.velocity)
                    if max_velocity < 0.01:  # Velocity threshold for "stopped"
                        self.moveit_client.get_logger().info("Movement completed (velocities near zero)")
                        return True
                
                # If no velocities available, use position change detection
                else:
                    # Store current position
                    current_positions = [joint_state.position[i] for i in range(len(joint_state.position))]
                    
                    # Wait a bit and check again
                    time.sleep(0.1)
                    rclpy.spin_once(self.moveit_client, timeout_sec=0.1)
                    
                    if self.moveit_client._current_joint_state is not None:
                        new_positions = [self.moveit_client._current_joint_state.position[i] 
                                       for i in range(len(self.moveit_client._current_joint_state.position))]
                        
                        # Check if positions have stopped changing
                        max_change = max(abs(new_positions[i] - current_positions[i]) 
                                       for i in range(min(len(new_positions), len(current_positions))))
                        
                        if max_change < 0.001:  # Position change threshold
                            self.moveit_client.get_logger().info("Movement completed (positions stabilized)")
                            return True
            
            # Spin once to update joint states
            rclpy.spin_once(self.moveit_client, timeout_sec=0.1)
            time.sleep(0.1)
        
        self.moveit_client.get_logger().warn(f"Movement completion timeout after {timeout} seconds")
        return False
    
    def move_to_joint_positions(self, joint_positions, velocity_scaling=None, acceleration_scaling=None):
        """
        Move to specific joint positions
        :param joint_positions: Dictionary of joint_name: position (in radians)
        :param velocity_scaling: Optional velocity scaling factor
        :param acceleration_scaling: Optional acceleration scaling factor
        """
        if velocity_scaling is None:
            velocity_scaling = self.default_velocity_scaling
        if acceleration_scaling is None:
            acceleration_scaling = self.default_acceleration_scaling
            
        self.moveit_client.get_logger().info("=== Moving to Joint Positions ===")
        self.print_current_state("Before movement")
        
        success = self.moveit_client.move_to_joint_configuration(
            joint_positions, velocity_scaling, acceleration_scaling
        )
        
        if success:
            # Wait for movement to complete instead of fixed time
            self.wait_for_movement_complete()
            self.print_current_state("After movement")
        
        return success
    
    def move_to_joint_degrees(self, joint_positions_deg, velocity_scaling=None, acceleration_scaling=None):
        """
        Move to specific joint positions specified in degrees
        :param joint_positions_deg: Dictionary of joint_name: position (in degrees)
        """
        # Convert degrees to radians
        joint_positions_rad = {}
        for joint, degrees in joint_positions_deg.items():
            joint_positions_rad[joint] = np.radians(degrees)
        
        return self.move_to_joint_positions(joint_positions_rad, velocity_scaling, acceleration_scaling)
    
    def move_to_pose_preferred_frame(self, position, orientation_euler, 
                                   reference_frame="base_link", target_link="link_6",
                                   velocity_scaling=None, acceleration_scaling=None):
        """
        Move to a specific pose using preferred reference frame
        :param position: [x, y, z] position in preferred frame
        :param orientation_euler: [roll, pitch, yaw] in preferred frame (in radians)
        :param reference_frame: Reference frame for the pose
        :param target_link: Target link name
        :param velocity_scaling: Optional velocity scaling factor
        :param acceleration_scaling: Optional acceleration scaling factor
        """
        if velocity_scaling is None:
            velocity_scaling = self.default_velocity_scaling
        if acceleration_scaling is None:
            acceleration_scaling = self.default_acceleration_scaling
        
        # Convert from preferred frame to MoveIt internal frame
        moveit_position, moveit_orientation = self.from_preferred_frame(
            position, orientation_euler, reference_frame, "base_link"
        )
        
        # Create pose message for MoveIt
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.header.stamp = self.moveit_client.get_clock().now().to_msg()
        
        # Set position
        target_pose.pose.position.x = float(moveit_position[0])
        target_pose.pose.position.y = float(moveit_position[1])
        target_pose.pose.position.z = float(moveit_position[2])
        
        # Convert euler to quaternion
        quat = quaternion_from_euler(moveit_orientation[0], moveit_orientation[1], moveit_orientation[2])
        target_pose.pose.orientation.x = float(quat[0])
        target_pose.pose.orientation.y = float(quat[1])
        target_pose.pose.orientation.z = float(quat[2])
        target_pose.pose.orientation.w = float(quat[3])
        
        self.moveit_client.get_logger().info("=== Moving to Pose (Preferred Frame) ===")
        self.moveit_client.get_logger().info(f"Target position (preferred frame): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        self.moveit_client.get_logger().info(f"Target orientation (preferred frame RPY): [{orientation_euler[0]:.3f}, {orientation_euler[1]:.3f}, {orientation_euler[2]:.3f}] rad")
        self.moveit_client.get_logger().info(f"Converted to MoveIt position: [{moveit_position[0]:.3f}, {moveit_position[1]:.3f}, {moveit_position[2]:.3f}]")
        self.moveit_client.get_logger().info(f"Converted to MoveIt orientation: [{moveit_orientation[0]:.3f}, {moveit_orientation[1]:.3f}, {moveit_orientation[2]:.3f}] rad")
        
        self.print_current_state("Before movement")
        
        success = self.moveit_client.move_to_pose(
            target_pose, target_link, velocity_scaling, acceleration_scaling
        )
        
        if success:
            # Wait for movement to complete instead of fixed time
            self.wait_for_movement_complete()
            self.print_current_state("After movement")
        
        return success
    
    def move_to_pose(self, position, orientation_euler=None, orientation_quat=None, 
                     frame_id="base_link", target_link="link_6",
                     velocity_scaling=None, acceleration_scaling=None):
        """
        Move to a specific pose using MoveIt internal frame (original method for compatibility)
        :param position: [x, y, z] position in meters
        :param orientation_euler: [roll, pitch, yaw] in radians (optional)
        :param orientation_quat: [x, y, z, w] quaternion (optional)
        :param frame_id: Reference frame
        :param target_link: Target link name
        :param velocity_scaling: Optional velocity scaling factor
        :param acceleration_scaling: Optional acceleration scaling factor
        """
        if velocity_scaling is None:
            velocity_scaling = self.default_velocity_scaling
        if acceleration_scaling is None:
            acceleration_scaling = self.default_acceleration_scaling
        
        # Create pose message
        target_pose = PoseStamped()
        target_pose.header.frame_id = frame_id
        target_pose.header.stamp = self.moveit_client.get_clock().now().to_msg()
        
        # Set position
        target_pose.pose.position.x = float(position[0])
        target_pose.pose.position.y = float(position[1])
        target_pose.pose.position.z = float(position[2])
        
        # Set orientation
        if orientation_quat is not None:
            target_pose.pose.orientation.x = float(orientation_quat[0])
            target_pose.pose.orientation.y = float(orientation_quat[1])
            target_pose.pose.orientation.z = float(orientation_quat[2])
            target_pose.pose.orientation.w = float(orientation_quat[3])
        elif orientation_euler is not None:
            quat = quaternion_from_euler(orientation_euler[0], orientation_euler[1], orientation_euler[2])
            target_pose.pose.orientation.x = float(quat[0])
            target_pose.pose.orientation.y = float(quat[1])
            target_pose.pose.orientation.z = float(quat[2])
            target_pose.pose.orientation.w = float(quat[3])
        else:
            # Default orientation (identity)
            target_pose.pose.orientation.w = 1.0
        
        self.moveit_client.get_logger().info("=== Moving to Pose (MoveIt Frame) ===")
        self.moveit_client.get_logger().info(f"Target position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        if orientation_euler is not None:
            self.moveit_client.get_logger().info(f"Target orientation (RPY): [{orientation_euler[0]:.3f}, {orientation_euler[1]:.3f}, {orientation_euler[2]:.3f}] rad")
        
        self.print_current_state("Before movement")
        
        success = self.moveit_client.move_to_pose(
            target_pose, target_link, velocity_scaling, acceleration_scaling
        )
        
        if success:
            # Wait for movement to complete instead of fixed time
            self.wait_for_movement_complete()
            self.print_current_state("After movement")
        
        return success
    
    def move_to_home(self):
        """Move to home position (all joints at 0)"""
        home_position = {
            'joint_1': 0.0,
            'joint_2': 0.0,
            'joint_3': 0.0,
            'joint_4': 0.0,
            'joint_5': 0.0,
            'joint_6': 0.0
        }
        self.moveit_client.get_logger().info("=== Moving to Home Position ===")
        return self.move_to_joint_positions(home_position)
    
    def set_velocity_scaling(self, scaling):
        """Set default velocity scaling factor (0.0 to 1.0)"""
        self.default_velocity_scaling = max(0.0, min(1.0, scaling))
        self.moveit_client.get_logger().info(f"Velocity scaling set to: {self.default_velocity_scaling}")
    
    def set_acceleration_scaling(self, scaling):
        """Set default acceleration scaling factor (0.0 to 1.0)"""
        self.default_acceleration_scaling = max(0.0, min(1.0, scaling))
        self.moveit_client.get_logger().info(f"Acceleration scaling set to: {self.default_acceleration_scaling}")
    
    def wait(self, seconds):
        """Wait for specified number of seconds (kept for manual delays if needed)"""
        self.moveit_client.get_logger().info(f"Waiting {seconds} seconds...")
        time.sleep(seconds)
    
    def shutdown(self):
        """Shutdown the robot interface"""
        self.moveit_client.destroy_node()
        rclpy.shutdown()

def main():
    """Example usage of the AR4Robot class with movement completion detection"""
    try:
        # Create robot interface
        robot = AR4Robot()
        
        # Print initial state in preferred frame
        robot.print_current_state("Initial robot state", reference_frame="base_link")
        
        # Example 1: Move to home position
        robot.move_to_home()
        
        # Example 2: Move to specific joint positions (using degrees)
        target_joints_deg = {
            'joint_1': 30.0,   # 30 degrees
            'joint_2': -30.0,  # -45 degrees  
            'joint_3': 30.0,   # 60 degrees
            'joint_4': 0.0,
            'joint_5': 30.0,   # 45 degrees
            'joint_6': 0.0
        }
        robot.move_to_joint_degrees(target_joints_deg)
        
        # Example 3: Move to pose using preferred reference frame
        try:
            target_position_preferred = [0.3, 0.0, 0.4]  # x, y, z in preferred frame
            target_orientation_preferred = [np.pi/2, 0.0, 0.0]  # roll, pitch, yaw in preferred frame
            robot.move_to_pose_preferred_frame(
                target_position_preferred, 
                target_orientation_preferred, 
                reference_frame="base_link"
            )
        except Exception as e:
            robot.moveit_client.get_logger().warn(f"Preferred frame pose movement failed: {e}")
        
        # Example 4: Move to another joint configuration
        target_joints_deg2 = {
            'joint_1': -30.0,
            'joint_2': 45.0,
            'joint_3': -60.0,
            'joint_4': 0.0,
            'joint_5': -45.0,
            'joint_6': 0.0
        }
        robot.move_to_joint_degrees(target_joints_deg2)
        
        # Return to home
        robot.move_to_home()
        
        # Keep alive
        robot.moveit_client.get_logger().info("Demo completed. Press Ctrl+C to exit.")
        rclpy.spin(robot.moveit_client)
        
    except KeyboardInterrupt:
        robot.moveit_client.get_logger().info("Shutting down...")
    finally:
        robot.shutdown()

if __name__ == '__main__':
    main()