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
from moveit_msgs.srv import GetPositionIK

from moveit_msgs.srv import GetPlanningScene, GetStateValidity, GetPositionFK
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import time
import numpy as np
import math

class MoveItActionClient(Node):
    def __init__(self, enable_logging=True):
        super().__init__('moveitaction_client')
        
        self.logging_enabled = enable_logging
        self.action_client = ActionClient(self, MoveGroup, '/move_action')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        
        # Subscribe to joint states
        self._joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.current_joint_state = None
        
        # Wait for the action server
        self.log_info("Waiting for move_group action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("move_group action server not available!")
            raise RuntimeError("move_group action server not available!")
                
        # Group name - this should match your SRDF
        self.group_name = "ar_manipulator"
        
        while self.current_joint_state is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.log_info("Joint states received")
    
    def log_info(self, message):
        if self.logging_enabled:
            self.get_logger().info(message)
    
    def log_warn(self, message):
        self.get_logger().warn(message)
    
    def log_error(self, message):
        self.get_logger().error(message)
    
    def enable_logging(self):
        self.logging_enabled = True

    def joint_state_callback(self, msg):
        self.current_joint_state = msg
        
    def get_current_joint_values(self):
        return self.current_joint_state

    def get_end_effector_pose(self, link_name="link_6"):
        """Get the current end effector pose using forward kinematics"""
        if self.current_joint_state is None:
            return None
            
        try:
            # Wait for FK service
            if not self.fk_client.wait_for_service(timeout_sec=2.0):
                self.log_warn("FK service not available")
                return None
            
            # Create FK request
            fk_request = GetPositionFK.Request()
            fk_request.header.frame_id = "base_link"
            fk_request.fk_link_names = [link_name]
            fk_request.robot_state.joint_state = self.current_joint_state
            
            # Call FK service
            future = self.fk_client.call_async(fk_request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.done():
                response = future.result()
                if response.error_code.val == 1 and len(response.pose_stamped) > 0:  # SUCCESS
                    return response.pose_stamped[0]
                else:
                    self.log_warn(f"FK computation failed with error code: {response.error_code.val}")
            else:
                self.log_warn("FK service call timed out")
                
        except Exception as e:
            self.log_warn(f"Error getting end effector pose: {e}")
            
        return None
    
    def get_ik(self,position, euler_angles=None, 
               frame_id="base_link", target_link="link_6", avoid_collisions=True):
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
        
        quat = quaternion_from_euler(
                float(euler_angles[0]), 
                float(euler_angles[1]), 
                float(euler_angles[2])
            )
        quat_xyzw = np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])
        
        # Create IK service client if it doesn't exist
        if not hasattr(self, '_ik_client'):
            self._ik_client = self.create_client(
                GetPositionIK, 
                'compute_ik'
            )
        
        # Wait for service to become available
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self.log_error("IK service not available")
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
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose.position.x = float(position[0])
        request.ik_request.pose_stamped.pose.position.y = float(position[1])
        request.ik_request.pose_stamped.pose.position.z = float(position[2])
        request.ik_request.pose_stamped.pose.orientation.x = float(quat_xyzw[0])
        request.ik_request.pose_stamped.pose.orientation.y = float(quat_xyzw[1])
        request.ik_request.pose_stamped.pose.orientation.z = float(quat_xyzw[2])
        request.ik_request.pose_stamped.pose.orientation.w = float(quat_xyzw[3])
        
        
        request.ik_request.robot_state.is_diff = True
        # Set robot state (use current joint state as seed)
        current_joint_state = self.get_current_joint_values()
            
        #joint_state_msg = JointState()
        #joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        #joint_state_msg.name = list(current_joint_state.keys())
        #joint_state_msg.position = list(current_joint_state.values())
        
        request.ik_request.robot_state.joint_state = current_joint_state
    
        # Call the service
        try:
            future = self._ik_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if future.result() is not None:
                response = future.result()
                if response.error_code.val == response.error_code.SUCCESS:

                    return np.array([response.solution.joint_state.position]).flatten()[:6]
                else:
                    self.log_warn(f"IK service failed with error code: {response.error_code.val}")
                    return None
            else:
                self.log_error("IK service call failed")
                return None
                
        except Exception as e:
            self.log_error(f"IK service call exception: {e}")
            return None
        

    def move_to_joint_configuration(self, joint_positions, velocity_scaling=1.0, acceleration_scaling=1.0):
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
            safety_margin = 0.05
            
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
                        self.log_error(f"Joint {joint_name} target position {target_pos:.4f} rad  "
                                    f"is outside safe bounds [{safe_min:.4f}, {safe_max:.4f}] rad ")
                        bounds_violated = True

            if bounds_violated:
                self.log_error("Movement rejected due to joint limit violations!")
                return False

            goal_msg = MoveGroup.Goal()
            
            # Create motion plan request
            req = MotionPlanRequest()
            req.group_name = self.group_name
            req.num_planning_attempts = 10
            req.max_velocity_scaling_factor = velocity_scaling
            req.max_acceleration_scaling_factor = acceleration_scaling
            req.allowed_planning_time = 10.0
            
            req.planner_id = "PTP"
            
            # Set joint constraints
            joint_constraints = []
            for joint_name, position in joint_positions_dict.items():
                constraint = JointConstraint()
                constraint.joint_name = joint_name
                constraint.position = position
                constraint.tolerance_above = 0.0001
                constraint.tolerance_below = 0.0001
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
            self.log_info("Sending goal to move_group...")
            future = self.action_client.send_goal_async(goal_msg)
            
            # Wait for the goal to be accepted
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.log_error("Goal rejected!")
                return False
            
            self.log_info("Goal accepted, waiting for result...")
            
            # Wait for the result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            
            result = result_future.result()
            if result and result.result.error_code.val == 1:  # SUCCESS
                self.log_info("Movement completed successfully!")
                return True
            else:
                error_code = result.result.error_code.val if result else "Unknown"
                self.log_error(f"Movement failed with error code: {error_code}")
                return False
                
        except Exception as e:
            self.log_error(f"Error in move_to_joint_configuration: {str(e)}")
            return False

    def move_to_pose(self, target_pose, target_link="link_6", velocity_scaling=1.0, acceleration_scaling=1.0):
        pos = target_pose.pose.position
        orient = target_pose.pose.orientation
        quat = [orient.x, orient.y, orient.z, orient.w]
        roll, pitch, yaw = euler_from_quaternion(quat)
        
        self.log_info(f"Target pose ({target_link}):")
        self.log_info(f"  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f} m")
        self.log_info(f"  Orientation (RPY): roll={roll:.3f} ({np.degrees(roll):.1f}°), "
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
        if self.current_joint_state:
            start_state = RobotState()
            start_state.joint_state = self.current_joint_state
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
        self.log_info("Sending pose goal to move_group...")
        future = self.action_client.send_goal_async(goal_msg)
        
        # Wait for the goal to be accepted
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_error("Goal rejected!")
            return False
        
        self.log_info("Goal accepted, waiting for result...")
        
        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        
        result = result_future.result()
        if result and result.result.error_code.val == 1:  # SUCCESS
            self.log_info("Movement completed successfully!")
            return True
        else:
            error_code = result.result.error_code.val if result else "Unknown"
            self.log_error(f"Movement failed with error code: {error_code}")
            return False

def main(args=None):
    rclpy.init()
    
    moveit_client = MoveItActionClient()
    
    
    movement = np.array([0.5,0.5,0.5,0.5,0.5,0.5])  
    success = moveit_client.move_to_joint_configuration(movement)
    if success:
        moveit_client.log_info(f"Movement completed successfully!")
    else:
        moveit_client.log_error(f"Movement failed!")
    
    
    rclpy.spin(moveit_client)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()