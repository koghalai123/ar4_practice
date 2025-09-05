#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_action_client import MoveItActionClient
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf_transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix, euler_from_matrix
import numpy as np
import time
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

def create_transformation_matrix(translation, euler_angles):
    """Create a 4x4 homogeneous transformation matrix."""
    transform = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')
    transform[:3, 3] = translation
    return transform

class AR4Robot(Node):
    def __init__(self):
        """Initialize the AR4 Robot interface"""
        
        
        # Logging control (enabled by default)
        self.logging_enabled = True
        self.warn_enabled = True
        
        # Create MoveIt client with logging setting
        self.moveit_client = MoveItActionClient(enable_logging=self.logging_enabled)
        
        # Default scaling factors for safety
        self.default_velocity_scaling = 1.0
        self.default_acceleration_scaling = 1.0
        
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
    
    def _log_info(self, message):
        """Centralized logging method that respects the logging_enabled setting"""
        if self.logging_enabled:
            self.moveit_client.get_logger().info(message)
    
    def _log_warn(self, message):
        """Centralized warning logging method (always outputs)"""
        if self.warn_enabled:
            self.moveit_client.get_logger().warn(message)
    
    def _log_error(self, message):
        """Centralized error logging method (always outputs)"""
        self.moveit_client.get_logger().error(message)
        
    def from_preferred_frame(self, position=None, euler_angles=None, old_reference_frame="base_link", new_reference_frame="base_link"):
        """Convert from preferred reference frame to MoveIt internal frame"""
        
        
        
        if new_reference_frame == old_reference_frame:
            position = np.array([position[0], position[1], position[2], 1.0])
            roll = euler_angles[1] - self.angle_offsets["pitch"]
            pitch = euler_angles[0] - self.angle_offsets["roll"]
            yaw = euler_angles[2] - self.angle_offsets["yaw"]
        elif new_reference_frame == "end_effector_link" and old_reference_frame == "base_link":
            position = np.array([position[0], position[1], position[2], 1.0]) - np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"], 0.0])   
            roll = euler_angles[1] - self.angle_offsets["pitch"]
            pitch = euler_angles[0] - self.angle_offsets["roll"]
            yaw = euler_angles[2] - self.angle_offsets["yaw"]     
        elif new_reference_frame == "base_link" and old_reference_frame == "end_effector_link":
            position = np.array([position[0], position[1], position[2], 1.0]) + np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"], 0.0])
            roll = euler_angles[1] - self.angle_offsets["pitch"]
            pitch = euler_angles[0] - self.angle_offsets["roll"]
            yaw = euler_angles[2] - self.angle_offsets["yaw"]        
        elif new_reference_frame == "global" and old_reference_frame == "base_link":
            position = np.array([position[0], position[1], position[2], 1.0])
            roll = euler_angles[1]
            pitch = euler_angles[0]
            yaw = euler_angles[2]
        else:
            print("Warning: Unhandled reference frame conversion in ar4_robot_py.py.")
            return None, None

        transformed_position = (np.dot(self.inverse_transformation_matrix, position)[:3])
        
        
        
        transformed_orientation_matrix = np.dot(self.inverse_transformation_matrix[:3, :3], euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3])
        transformed_orientation = euler_from_matrix(transformed_orientation_matrix, axes='sxyz')

        return transformed_position, transformed_orientation

    '''def to_preferred_frame(self, position=None, euler_angles=None, new_reference_frame="base_link"):
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
        if new_reference_frame == "end_effector_link":
            transformed_position = transformed_position - np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"]])
        elif new_reference_frame == "base_link":
            transformed_position = transformed_position + np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"]])

        return transformed_position, transformed_orientation'''
    def to_preferred_frame(self, position=None, euler_angles=None, old_reference_frame="base_link", new_reference_frame="base_link"):
        """Convert from MoveIt internal frame to preferred reference frame"""
        # Apply transformation matrix to position
        position_homogeneous = np.array([position[0], position[1], position[2], 1.0])
        transformed_position = np.dot(self.transformation_matrix, position_homogeneous)[:3]
        # FIRST: Apply matrix transformation to orientation (inverse of from_preferred_frame step 2)
        orientation_matrix = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')[:3, :3]
        transformed_orientation_matrix = np.dot(self.transformation_matrix[:3, :3], orientation_matrix)
        transformed_euler = euler_from_matrix(transformed_orientation_matrix, axes='sxyz')
        
        # SECOND: Apply angle offsets (inverse of from_preferred_frame step 1)
        # Note: from_preferred_frame SUBTRACTS offsets, so to_preferred_frame must ADD them
        # Also note the index mapping: from_preferred_frame maps [0,1,2] -> [1,0,2], so inverse maps [0,1,2] -> [1,0,2]
        pitch = transformed_euler[1] + self.angle_offsets["pitch"]  # transformed_euler[1] becomes pitch (index 0 in output)
        roll = transformed_euler[0] + self.angle_offsets["roll"]    # transformed_euler[0] becomes roll (index 1 in output)  
        yaw = transformed_euler[2] + self.angle_offsets["yaw"]      # transformed_euler[2] becomes yaw (index 2 in output)
        
        transformed_orientation = np.array([pitch, roll, yaw])
        
        if new_reference_frame != old_reference_frame:
            if new_reference_frame == "end_effector_link" and old_reference_frame == "base_link":
                transformed_position = transformed_position - np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"]])
            elif new_reference_frame == "base_link" and old_reference_frame == "end_effector_link":
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
    
    def move_to_joint_positions(self, joint_positions, velocity_scaling=1.0, acceleration_scaling=1.0):
        """
        Move to specific joint positions
        :param joint_positions: Dictionary of joint_name: position (in radians)
        :param velocity_scaling: Optional velocity scaling factor
        :param acceleration_scaling: Optional acceleration scaling factor
        """
                    
        success = self.moveit_client.move_to_joint_configuration(
            joint_positions, velocity_scaling, acceleration_scaling
        )
        
        return success
    
    '''def move_to_joint_degrees(self, joint_positions_deg, velocity_scaling=None, acceleration_scaling=None):
        """
        Move to specific joint positions specified in degrees
        :param joint_positions_deg: Dictionary of joint_name: position (in degrees)
        """
        # Convert degrees to radians
        joint_positions_rad = {}
        for joint, degrees in joint_positions_deg.items():
            joint_positions_rad[joint] = np.radians(degrees)
        
        return self.move_to_joint_positions(joint_positions_rad, velocity_scaling, acceleration_scaling)'''
    
    def move_to_pose_preferred_frame(self, position, orientation_euler, 
                                   frame_id="base_link", target_link="link_6",
                                   velocity_scaling=1.0, acceleration_scaling=1.0):
        """
        Move to a specific pose using preferred reference frame
        :param position: [x, y, z] position in preferred frame
        :param orientation_euler: [roll, pitch, yaw] in preferred frame (in radians)
        :param reference_frame: Reference frame for the pose
        :param target_link: Target link name
        :param velocity_scaling: Optional velocity scaling factor
        :param acceleration_scaling: Optional acceleration scaling factor
        """
        # Convert from preferred frame to MoveIt internal frame
        moveit_position, moveit_orientation = self.from_preferred_frame(
            position, orientation_euler, frame_id, "base_link"
        )
        success = self.move_to_pose(position=moveit_position, orientation_euler=moveit_orientation, 
                     frame_id=frame_id, target_link=target_link, velocity_scaling=velocity_scaling, acceleration_scaling=acceleration_scaling)

        return success
    
    def move_to_pose(self, position, orientation_euler=None, orientation_quat=None, 
                     frame_id="base_link", target_link="link_6", velocity_scaling=1.0,
                     acceleration_scaling = 1.0):
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
        target_pose.header.frame_id = "base_link"
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
            self._log_warn("No orientation provided")
            return False
        
        success = self.moveit_client.move_to_pose(
            target_pose, target_link, velocity_scaling, acceleration_scaling
        )
        return success
    
    def move_to_home(self):
        """Move to home position (all joints at 0)"""
        home_position = np.zeros(6)
        return self.move_to_joint_positions(home_position)
    
    def enable_logging(self):
        """Enable informational logging"""
        self.logging_enabled = True
        self.moveit_client.logging_enabled = True
        self.moveit_client.get_logger().info("Logging enabled")
    
    def disable_logging(self):
        """Disable informational logging (warnings and errors still shown)"""
        if self.logging_enabled:
            self.moveit_client.get_logger().info("Logging disabled")
        self.logging_enabled = False
        self.moveit_client.logging_enabled = False
    
    def shutdown(self):
        """Shutdown the robot interface"""
        self.moveit_client.destroy_node()
        rclpy.shutdown()
    
    def _call_ik_service(self, position, quat_xyzw, frame_id="base_link", 
                         target_link="link_6", avoid_collisions=True):
        """
        Call MoveIt's IK service directly to compute inverse kinematics.
        
        Parameters:
        - position: [x, y, z] position in meters
        - quat_xyzw: [x, y, z, w] quaternion
        - frame_id: Reference frame for the pose
        - target_link: Target link name
        - avoid_collisions: Whether to avoid collisions during IK calculation
        
        Returns:
        - JointState message with the solution, or None if no solution found
        """
        # Create IK service client if it doesn't exist
        if not hasattr(self, '_ik_client'):
            self._ik_client = self.moveit_client.create_client(
                GetPositionIK, 
                'compute_ik'
            )
        
        # Wait for service to become available
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self._log_error("IK service not available")
            return None
        
        # Create the request
        request = GetPositionIK.Request()
        
        # Set up the IK request
        request.ik_request.group_name = "ar_manipulator"
        request.ik_request.ik_link_name = target_link
        request.ik_request.avoid_collisions = avoid_collisions
        
        # THIS CANNOT BE A FLOAT OR IT CAUSES A VARIABLE TYPE ERROR. NO DECIMAL POINTS
        request.ik_request.timeout.sec = 0
        request.ik_request.timeout.nanosec = 100000000

        
        # Set the target pose
        request.ik_request.pose_stamped.header.frame_id = frame_id
        request.ik_request.pose_stamped.header.stamp = self.moveit_client.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose.position.x = float(position[0])
        request.ik_request.pose_stamped.pose.position.y = float(position[1])
        request.ik_request.pose_stamped.pose.position.z = float(position[2])
        request.ik_request.pose_stamped.pose.orientation.x = float(quat_xyzw[0])
        request.ik_request.pose_stamped.pose.orientation.y = float(quat_xyzw[1])
        request.ik_request.pose_stamped.pose.orientation.z = float(quat_xyzw[2])
        request.ik_request.pose_stamped.pose.orientation.w = float(quat_xyzw[3])
        
        
        request.ik_request.robot_state.is_diff = True
        # Set robot state (use current joint state as seed)
        current_joint_state = self.get_current_joint_state()
        if current_joint_state:
            
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.moveit_client.get_clock().now().to_msg()
            joint_state_msg.name = list(current_joint_state.keys())
            joint_state_msg.position = list(current_joint_state.values())
            
            request.ik_request.robot_state.joint_state = joint_state_msg
        
        # Call the service
        try:
            future = self._ik_client.call_async(request)
            rclpy.spin_until_future_complete(self.moveit_client, future, timeout_sec=10.0)
            
            if future.result() is not None:
                response = future.result()
                if response.error_code.val == response.error_code.SUCCESS:
                    return response.solution.joint_state
                else:
                    self._log_warn(f"IK service failed with error code: {response.error_code.val}")
                    return None
            else:
                self._log_error("IK service call failed")
                return None
                
        except Exception as e:
            self._log_error(f"IK service call exception: {e}")
            return None
        

    def get_fk(self, joint_positions, target_link="link_6", frame_id="base_link"):
        """
        Calculate forward kinematics for given joint positions.
        
        Parameters:
        - joint_positions: Dictionary mapping joint names to angles (in radians)
                        OR numpy array of 6 joint angles [joint_1, joint_2, ..., joint_6]
        - target_link: Target link name (default "link_6" for end effector)
        - frame_id: Reference frame for the returned pose
        
        Returns:
        - Tuple (position, euler_angles) where:
        * position: [x, y, z] position in meters in the specified frame
        * euler_angles: [roll, pitch, yaw] in radians in the specified frame
        - Returns (None, None) if FK calculation fails
        """
        try:
            # Handle different input formats
            if isinstance(joint_positions, np.ndarray):
                # Convert numpy array to dictionary
                if len(joint_positions) != 6:
                    self._log_error("Joint positions array must have 6 elements")
                    return None, None
                joint_dict = {
                    f'joint_{i+1}': float(joint_positions[i]) 
                    for i in range(6)
                }
            elif isinstance(joint_positions, dict):
                joint_dict = joint_positions
            else:
                self._log_error("joint_positions must be dict or numpy array")
                return None, None

            # Create FK service client if it doesn't exist
            if not hasattr(self, '_fk_client'):
                self._fk_client = self.moveit_client.create_client(
                    GetPositionFK, 
                    'compute_fk'
                )
            
            # Wait for service to become available
            if not self._fk_client.wait_for_service(timeout_sec=5.0):
                self._log_error("FK service not available")
                return None, None
            
            # Create the request
            request = GetPositionFK.Request()
            
            # Set up the FK request
            request.fk_link_names = [target_link]
            request.header.frame_id = "base_link"  # Always compute in base_link first
            request.header.stamp = self.moveit_client.get_clock().now().to_msg()
            
            # Create robot state with specified joint positions
            robot_state = RobotState()
            joint_state = JointState()
            joint_state.header.stamp = self.moveit_client.get_clock().now().to_msg()
            
            # Set joint positions
            joint_state.name = list(joint_dict.keys())
            joint_state.position = list(joint_dict.values())
            
            robot_state.joint_state = joint_state
            request.robot_state = robot_state
            
            # Call the service
            future = self._fk_client.call_async(request)
            rclpy.spin_until_future_complete(self.moveit_client, future, timeout_sec=10.0)
            
            if future.result() is not None:
                response = future.result()
                if response.error_code.val == response.error_code.SUCCESS:
                    if len(response.pose_stamped) > 0:
                        pose_stamped = response.pose_stamped[0]
                        
                        # Extract position
                        pos = pose_stamped.pose.position
                        position = np.array([pos.x, pos.y, pos.z])
                        
                        # Extract orientation and convert to euler
                        orient = pose_stamped.pose.orientation
                        quat = [orient.x, orient.y, orient.z, orient.w]
                        euler_angles = np.array(euler_from_quaternion(quat))
                        
                        # Convert from MoveIt's base_link frame to desired frame
                        position_preferred, euler_angles_preferred = self.to_preferred_frame(
                            position=position, euler_angles=euler_angles, new_reference_frame=frame_id
                        )

                        return position_preferred, euler_angles_preferred
                    else:
                        self._log_error("FK service returned empty pose list")
                        return None, None
                else:
                    self._log_error(f"FK service failed with error code: {response.error_code.val}")
                    return None, None
            else:
                self._log_error("FK service call failed")
                return None, None
                
        except Exception as e:
            self._log_error(f"FK calculation exception: {e}")
            return None, None
    
    def get_ik(self, position, euler_angles=None, 
               frame_id="base_link", target_link="link_6", avoid_collisions=True):
        """
        Calculate inverse kinematics for a specific pose.
        
        Parameters:
        - position: [x, y, z] position in meters
        - orientation_euler: [roll, pitch, yaw] in radians (optional)
        - orientation_quat: [x, y, z, w] quaternion (optional)
        - frame_id: Reference frame for the pose
        - target_link: Target link name
        - avoid_collisions: Whether to avoid collisions during IK calculation
        
        Returns:
        - Dictionary mapping joint names to angles (in radians) if solution found
        - None if no IK solution found
        """
                
        # Convert from input frame to MoveIt's base_link frame if needed
        transformed_position, transformed_orientation = self.from_preferred_frame(position, euler_angles, old_reference_frame=frame_id, new_reference_frame="base_link")
        quat = quaternion_from_euler(
                float(transformed_orientation[0]), 
                float(transformed_orientation[1]), 
                float(transformed_orientation[2])
            )
        quat_xyzw = np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])

        joint_state = self._call_ik_service(
            position=transformed_position,
            quat_xyzw=quat_xyzw,
            frame_id="base_link",  # Always use base_link for MoveIt
            target_link=target_link,
            avoid_collisions=avoid_collisions
        )
        
        if joint_state is None:
            self._log_warn("No IK solution found")
            return None
        
        # Convert joint state message to dictionary
        joint_positions = {}
        for i, name in enumerate(joint_state.name):
            # Only include joints that are part of the manipulator group
            if name.startswith('joint_'):
                joint_positions[name] = joint_state.position[i]
        
        if not joint_positions:
            self._log_warn("IK solution found but no valid joint positions")
            return None
        
        self._log_info("IK solution found:")
        for joint, value in joint_positions.items():
            self._log_info(f"  {joint}: {value:.3f} rad ({np.degrees(value):.1f}°)")
        
        joint_array = np.array(list(joint_positions.values()))
        
        return joint_array

    
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