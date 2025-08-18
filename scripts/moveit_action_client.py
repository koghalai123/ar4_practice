#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState
)
from moveit_msgs.srv import GetPlanningScene, GetStateValidity, GetPositionFK
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion
import time
import numpy as np
import math

class MoveItActionClient(Node):
    def __init__(self, enable_logging=True):
        super().__init__('moveit_action_client')
        
        # Logging control
        self.logging_enabled = enable_logging
        
        # Create action client for move_group
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        
        # Create service clients
        self._planning_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk')
        
        # Subscribe to joint states
        self._joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        
        self._current_joint_state = None
        
        # Wait for the action server
        self._log_info("Waiting for move_group action server...")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("move_group action server not available!")
            raise RuntimeError("move_group action server not available!")
        
        self._log_info("Connected to move_group action server")
        
        # Group name - this should match your SRDF
        self.group_name = "ar_manipulator"
        
        # Wait for joint states
        self._log_info("Waiting for joint states...")
        while self._current_joint_state is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        self._log_info("Joint states received")
    
    def _log_info(self, message):
        """Centralized logging method that respects the logging_enabled setting"""
        if self.logging_enabled:
            self.get_logger().info(message)
    
    def _log_warn(self, message):
        """Centralized warning logging method (always outputs)"""
        self.get_logger().warn(message)
    
    def _log_error(self, message):
        """Centralized error logging method (always outputs)"""
        self.get_logger().error(message)
    
    def enable_logging(self):
        """Enable informational logging"""
        self.logging_enabled = True
    
    def disable_logging(self):
        """Disable informational logging (warnings and errors still shown)"""
        self.logging_enabled = False

    def _joint_state_callback(self, msg):
        """Callback to store current joint state"""
        self._current_joint_state = msg

    def get_current_joint_values(self):
        """Get current joint values as a dictionary"""
        if self._current_joint_state is None:
            return None
        
        joint_dict = {}
        for i, name in enumerate(self._current_joint_state.name):
            if i < len(self._current_joint_state.position):
                joint_dict[name] = self._current_joint_state.position[i]
        
        return joint_dict

    def get_end_effector_pose(self, link_name="link_6"):
        """Get the current end effector pose using forward kinematics"""
        if self._current_joint_state is None:
            return None
            
        try:
            # Wait for FK service
            if not self._fk_client.wait_for_service(timeout_sec=2.0):
                self._log_warn("FK service not available")
                return None
            
            # Create FK request
            fk_request = GetPositionFK.Request()
            fk_request.header.frame_id = "base_link"
            fk_request.fk_link_names = [link_name]
            fk_request.robot_state.joint_state = self._current_joint_state
            
            # Call FK service
            future = self._fk_client.call_async(fk_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.done():
                response = future.result()
                if response.error_code.val == 1 and len(response.pose_stamped) > 0:  # SUCCESS
                    return response.pose_stamped[0]
                else:
                    self._log_warn(f"FK computation failed with error code: {response.error_code.val}")
            else:
                self._log_warn("FK service call timed out")
                
        except Exception as e:
            self._log_warn(f"Error getting end effector pose: {e}")
            
        return None

    def print_robot_state(self, prefix="Robot state", link_name="link_6"):
        """Print both joint positions and end effector pose"""
        # Print joint positions
        joints = self.get_current_joint_values()
        if joints:
            self._log_info(f"{prefix} - Joint positions:")
            for joint, value in joints.items():
                self._log_info(f"  {joint}: {value:.3f} rad ({np.degrees(value):.1f}°)")
        else:
            self._log_warn("No joint state available")
            
        # Print end effector pose
        pose = self.get_end_effector_pose(link_name)
        if pose:
            pos = pose.pose.position
            orient = pose.pose.orientation
            
            # Convert quaternion to euler angles
            quat = [orient.x, orient.y, orient.z, orient.w]
            roll, pitch, yaw = euler_from_quaternion(quat)
            
            self._log_info(f"{prefix} - End effector pose ({link_name}):")
            self._log_info(f"  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f} m")
            self._log_info(f"  Orientation (RPY): roll={roll:.3f} ({np.degrees(roll):.1f}°), "
                                 f"pitch={pitch:.3f} ({np.degrees(pitch):.1f}°), "
                                 f"yaw={yaw:.3f} ({np.degrees(yaw):.1f}°)")
            self._log_info(f"  Orientation (Quat): x={orient.x:.3f}, y={orient.y:.3f}, "
                                 f"z={orient.z:.3f}, w={orient.w:.3f}")
        else:
            self._log_warn("Could not get end effector pose")

        
    def reset_planning_scene(self):
        """Reset the planning scene to clear any cached states"""
        try:
            if self._planning_scene_client.wait_for_service(timeout_sec=2.0):
                req = GetPlanningScene.Request()
                req.components.components = req.components.SCENE_SETTINGS
                future = self._planning_scene_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                self._log_info("Planning scene reset")
            else:
                self._log_warn("Planning scene service not available")
        except Exception as e:
            self._log_warn(f"Could not reset planning scene: {e}")

    def move_to_joint_configuration(self, joint_positions, velocity_scaling=1.0, acceleration_scaling=1.0):
        """
        Move robot to specified joint configuration
        :param joint_positions: Dictionary of joint_name: position
        :param velocity_scaling: Velocity scaling factor (0.0 to 1.0)
        :param acceleration_scaling: Acceleration scaling factor (0.0 to 1.0)
        """
        try:
            # Define ACTUAL joint limits from mk3.yaml (converted from degrees to radians)
            joint_limits = {
                'joint_1': (math.radians(-170), math.radians(170)),   # -2.967 to 2.967 rad
                'joint_2': (math.radians(-42), math.radians(90)),     # -0.733 to 1.571 rad  
                'joint_3': (math.radians(-89), math.radians(52)),     # -1.553 to 0.908 rad
                'joint_4': (math.radians(-180), math.radians(180)),   # -3.142 to 3.142 rad
                'joint_5': (math.radians(-105), math.radians(105)),   # -1.833 to 1.833 rad
                'joint_6': (math.radians(-200), math.radians(200))    # -3.142 to 3.142 rad
            }
            
            # Safety margin from limits (in radians) - adjust as needed
            safety_margin = 0.01
            
            # Convert to dictionary if it's a list
            joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
            joint_positions_dict = {}
            for i, name in enumerate(joint_names):
                joint_positions_dict[name] = float(joint_positions[i])

            
            # Check bounds for each joint
            bounds_violated = False
            for joint_name, target_pos in joint_positions_dict.items():
                if joint_name in joint_limits:
                    min_limit, max_limit = joint_limits[joint_name]
                    safe_min = min_limit + safety_margin
                    safe_max = max_limit - safety_margin
                    
                    if target_pos < safe_min or target_pos > safe_max:
                        self._log_error(f"Joint {joint_name} target position {target_pos:.4f} rad  "
                                    f"is outside safe bounds [{safe_min:.4f}, {safe_max:.4f}] rad ")
                        '''self._log_error(f"  Actual limits: [{min_limit:.4f}, {max_limit:.4f}] rad "
                                    f"([{math.degrees(min_limit):.1f}°, {math.degrees(max_limit):.1f}°])")'''
                        bounds_violated = True
                    else:
                        # Log successful bounds check
                        self._log_info(f"Joint {joint_name}: {target_pos:.4f} rad ({math.degrees(target_pos):.1f}°) "
                                    f"is within safe bounds")
            
            if bounds_violated:
                #self._log_error("Movement rejected due to joint limit violations!")
                return False

            
            # Create the goal
            goal_msg = MoveGroup.Goal()
            
            # Create motion plan request
            req = MotionPlanRequest()
            req.group_name = self.group_name
            req.num_planning_attempts = 10
            req.max_velocity_scaling_factor = velocity_scaling
            req.max_acceleration_scaling_factor = acceleration_scaling
            req.allowed_planning_time = 10.0
            
            req.planner_id = "PTP"
            
            '''# Set start state to current state
            if self._current_joint_state:
                start_state = RobotState()
                start_state.joint_state = self._current_joint_state
                req.start_state = start_state'''
            
            
            

            # Set joint constraints
            joint_constraints = []
            for joint_name, position in joint_positions_dict.items():
                constraint = JointConstraint()
                constraint.joint_name = joint_name
                constraint.position = position
                constraint.tolerance_above = 0.001
                constraint.tolerance_below = 0.001
                constraint.weight = 1.0
                joint_constraints.append(constraint)
            
            goal_constraint = Constraints()
            goal_constraint.joint_constraints = joint_constraints
            req.goal_constraints = [goal_constraint]
            
            # Set workspace bounds
            req.workspace_parameters.header.frame_id = "base_link"
            req.workspace_parameters.min_corner.x = -10.0
            req.workspace_parameters.min_corner.y = -10.0
            req.workspace_parameters.min_corner.z = -10.0
            req.workspace_parameters.max_corner.x = 10.0
            req.workspace_parameters.max_corner.y = 10.0
            req.workspace_parameters.max_corner.z = 10.0
            
            goal_msg.request = req
            
            # Planning options
            planning_options = PlanningOptions()
            planning_options.plan_only = False
            planning_options.look_around = False
            planning_options.look_around_attempts = 0
            planning_options.max_safe_execution_cost = 10.0
            planning_options.replan = True
            planning_options.replan_attempts = 10
            planning_options.replan_delay = 0.0
            
            goal_msg.planning_options = planning_options
            
            # Send the goal
            self._log_info("Sending goal to move_group...")
            future = self._action_client.send_goal_async(goal_msg)
            
            # Wait for the goal to be accepted
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self._log_error("Goal rejected!")
                return False
            
            self._log_info("Goal accepted, waiting for result...")
            
            # Wait for the result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            
            result = result_future.result()
            if result and result.result.error_code.val == 1:  # SUCCESS
                self._log_info("Movement completed successfully!")
                
                
                # Print final state
                self.print_robot_state("Final state after movement")
                
                return True
            else:
                error_code = result.result.error_code.val if result else "Unknown"
                self._log_error(f"Movement failed with error code: {error_code}")
                return False
                
        except Exception as e:
            self._log_error(f"Error in move_to_joint_configuration: {str(e)}")
            return False

    def move_to_pose(self, target_pose, target_link="link_6", velocity_scaling=1.0, acceleration_scaling=1.0):
        """
        Move robot to specified pose
        :param target_pose: geometry_msgs/PoseStamped target pose
        :param target_link: string name of target link
        :param velocity_scaling: Velocity scaling factor (0.0 to 1.0)
        :param acceleration_scaling: Acceleration scaling factor (0.0 to 1.0)
        """
        try:
            # Print current state before movement
            self.print_robot_state("Current state before pose movement", target_link)
            
            # Print target pose
            pos = target_pose.pose.position
            orient = target_pose.pose.orientation
            quat = [orient.x, orient.y, orient.z, orient.w]
            roll, pitch, yaw = euler_from_quaternion(quat)
            
            self._log_info(f"Target pose ({target_link}):")
            self._log_info(f"  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f} m")
            self._log_info(f"  Orientation (RPY): roll={roll:.3f} ({np.degrees(roll):.1f}°), "
                                 f"pitch={pitch:.3f} ({np.degrees(pitch):.1f}°), "
                                 f"yaw={yaw:.3f} ({np.degrees(yaw):.1f}°)")
            
            # Create the goal
            goal_msg = MoveGroup.Goal()
            
            # Create motion plan request
            req = MotionPlanRequest()
            req.group_name = self.group_name
            req.num_planning_attempts = 10
            req.max_velocity_scaling_factor = velocity_scaling
            req.max_acceleration_scaling_factor = acceleration_scaling
            req.allowed_planning_time = 2.0
            
            # Set start state to current state
            if self._current_joint_state:
                start_state = RobotState()
                start_state.joint_state = self._current_joint_state
                req.start_state = start_state
            
            # Position constraint
            position_constraint = PositionConstraint()
            position_constraint.header = target_pose.header
            position_constraint.link_name = target_link
            position_constraint.target_point_offset.x = 0.0
            position_constraint.target_point_offset.y = 0.0
            position_constraint.target_point_offset.z = 0.0
            
            # Constraint region (small box around target)
            constraint_region = BoundingVolume()
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.01, 0.01, 0.01]  # 1cm tolerance
            constraint_region.primitives = [box]
            constraint_region.primitive_poses = [target_pose.pose]
            position_constraint.constraint_region = constraint_region
            position_constraint.weight = 1.0
            
            # Orientation constraint
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header = target_pose.header
            orientation_constraint.link_name = target_link
            orientation_constraint.orientation = target_pose.pose.orientation
            orientation_constraint.absolute_x_axis_tolerance = 0.02
            orientation_constraint.absolute_y_axis_tolerance = 0.02
            orientation_constraint.absolute_z_axis_tolerance = 0.02
            orientation_constraint.weight = 1.0
            
            # Combine constraints
            goal_constraint = Constraints()
            goal_constraint.position_constraints = [position_constraint]
            goal_constraint.orientation_constraints = [orientation_constraint]
            req.goal_constraints = [goal_constraint]
            
            # Set workspace bounds
            req.workspace_parameters.header.frame_id = target_link#target_pose.header.frame_id
            req.workspace_parameters.min_corner.x = -1.0
            req.workspace_parameters.min_corner.y = -1.0
            req.workspace_parameters.min_corner.z = -1.0
            req.workspace_parameters.max_corner.x = 1.0
            req.workspace_parameters.max_corner.y = 1.0
            req.workspace_parameters.max_corner.z = 1.0
            
            goal_msg.request = req
            
            # Planning options
            planning_options = PlanningOptions()
            planning_options.plan_only = False
            planning_options.look_around = False
            planning_options.look_around_attempts = 0
            planning_options.max_safe_execution_cost = 10.0
            planning_options.replan = True
            planning_options.replan_attempts = 10
            planning_options.replan_delay = 0.0
            
            goal_msg.planning_options = planning_options
            
            # Send the goal
            self._log_info("Sending pose goal to move_group...")
            future = self._action_client.send_goal_async(goal_msg)
            
            # Wait for the goal to be accepted
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self._log_error("Goal rejected!")
                return False
            
            self._log_info("Goal accepted, waiting for result...")
            
            # Wait for the result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            
            result = result_future.result()
            if result and result.result.error_code.val == 1:  # SUCCESS
                self._log_info("Movement completed successfully!")
                
                # Print final state
                self.print_robot_state("Final state after pose movement", target_link)
                
                return True
            else:
                error_code = result.result.error_code.val if result else "Unknown"
                self._log_error(f"Movement failed with error code: {error_code}")
                return False
                
        except Exception as e:
            self._log_error(f"Error in move_to_pose: {str(e)}")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    # Create the node
    moveit_client = MoveItActionClient()
    
    try:
        # Print initial state
        moveit_client.print_robot_state("Initial robot state")
        
        # Multiple sequential movements
        movements = [
            {
                'type': 'joint',
                'values': {
                    'joint_1': 0.0,
                    'joint_2': 0.0,
                    'joint_3': 0.0,
                    'joint_4': 0.0,
                    'joint_5': 0.0,
                    'joint_6': 0.0
                }
            },
            {
                'type': 'joint',
                'values': {
                    'joint_1': 0.5,
                    'joint_2': -0.5,
                    'joint_3': 0.5,
                    'joint_4': 0.0,
                    'joint_5': 0.5,
                    'joint_6': 0.0
                }
            },
            {
                'type': 'joint',
                'values': {
                    'joint_1': -0.5,
                    'joint_2': 0.5,
                    'joint_3': -0.5,
                    'joint_4': 0.0,
                    'joint_5': -0.5,
                    'joint_6': 0.0
                }
            }
        ]
        
        for i, movement in enumerate(movements):
            moveit_client._log_info(f"--- Movement {i+1} ---")
            
            if movement['type'] == 'joint':
                success = moveit_client.move_to_joint_configuration(movement['values'])
                if success:
                    moveit_client._log_info(f"Movement {i+1} completed successfully!")
                else:
                    moveit_client._log_error(f"Movement {i+1} failed!")
                    break
            
            # Wait between movements
            if i < len(movements) - 1:
                moveit_client._log_info("Waiting before next movement...")
                time.sleep(2.0)
        
        # Keep the node alive
        moveit_client._log_info("All movements completed. Press Ctrl+C to exit.")
        rclpy.spin(moveit_client)
        
    except KeyboardInterrupt:
        moveit_client._log_info("Shutting down...")
    
    moveit_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()