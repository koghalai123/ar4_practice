#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from tf_transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix, euler_from_matrix
from moveit_configs_utils import MoveItConfigsBuilder
import time

def create_transformation_matrix(translation, euler_angles):
    """Create a 4x4 homogeneous transformation matrix."""
    transform = euler_matrix(eulerAngles[0], eulerAngles[1], eulerAngles[2], axes='sxyz')
    transform[:3, 3] = translation
    return transform

class AR4_ROBOT(Node):
    def __init__(self):
        super().__init__("ar4_robot_commander")
        
        # Load the configuration files for your robot. 
        # Replace "ar4_moveit_config" with the name of your actual MoveIt configuration package.
        moveit_config = MoveItConfigsBuilder(
            robot_name="ar4_robot",
            package_name="ar4_moveit_config" 
        ).to_moveit_configs()

        # Instantiate MoveItPy with your robot's configuration
        self.moveit_py = MoveItPy(node=self, config_dict=moveit_config.to_dict())

        # Get the planning component for your robot's arm group
        self.ar_manipulator = self.moveit_py.get_planning_component("ar_manipulator")
        self.get_logger().info("MoveItPy commander initialized for AR4 robot.")

        # Set planner parameters
        self.ar_manipulator.set_parameters({"max_velocity_scaling_factor": 1.5, "max_acceleration_scaling_factor": 3.0})

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

    def plan_and_execute(self, planning_component):
        """A helper function to plan and execute a motion."""
        self.get_logger().info("Planning trajectory...")
        plan_result = planning_component.plan()
        
        if plan_result:
            self.get_logger().info("Executing plan...")
            self.moveit_py.execute(plan_result.trajectory)
        else:
            self.get_logger().error("Planning failed.")
            
    def move_to_joint_positions(self, joint_positions):
        """Move to specific joint positions."""
        self.ar_manipulator.set_start_state_to_current_state()
        self.ar_manipulator.set_goal_state(joint_state=joint_positions)
        self.plan_and_execute(self.ar_manipulator)

    def move_pose(self, inputPosition, eulerAngles, reference_frame="base_link"):
        """Move to a specific pose."""
        
        transformed_position, transformed_orientation = self.fromMyPreferredFrame(
            inputPosition, eulerAngles, old_reference_frame=reference_frame, new_reference_frame="base_link"
        )
        
        quat = quaternion_from_euler(
            transformed_orientation[0], transformed_orientation[1], transformed_orientation[2]
        )
        
        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose.position.x = float(transformed_position[0])
        target_pose.pose.position.y = float(transformed_position[1])
        target_pose.pose.position.z = float(transformed_position[2])
        target_pose.pose.orientation.x = float(quat[0])
        target_pose.pose.orientation.y = float(quat[1])
        target_pose.pose.orientation.z = float(quat[2])
        target_pose.pose.orientation.w = float(quat[3])
        
        self.ar_manipulator.set_start_state_to_current_state()
        self.ar_manipulator.set_goal_state(pose_stamped=target_pose, pose_link="link_6")
        self.plan_and_execute(self.ar_manipulator)

    def get_current_pose(self, reference_frame="base_link"):
        """Get the current end effector pose."""
        
        current_state = self.moveit_py.get_current_state()
        current_pose = current_state.get_pose("link_6")
        
        quat = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        eulerAngles = np.array([roll, pitch, yaw])
        
        position = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
        position, orientation = self.toMyPreferredFrame(position, eulerAngles, reference_frame)
        
        return position, orientation
        
    def fromMyPreferredFrame(self, position, eulerAngles, old_reference_frame="base_link", new_reference_frame="base_link"):
        if new_reference_frame == old_reference_frame:
            position = np.array([position[0], position[1], position[2], 1.0])
        elif new_reference_frame == "end_effector_link" and old_reference_frame == "base_link":
            position = np.array([position[0], position[1], position[2], 1.0]) - np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"],0.0])        
        elif new_reference_frame == "base_link" and old_reference_frame == "end_effector_link":
            position = np.array([position[0], position[1], position[2], 1.0]) + np.array([self.pos_offsets["x"], self.pos_offsets["y"], self.pos_offsets["z"],0.0])        

        transformed_position = (np.dot(self.inverse_transformation_matrix, position)[:3])
        
        roll = eulerAngles[1] - self.angle_offsets["pitch"]
        pitch = eulerAngles[0] - self.angle_offsets["roll"]
        yaw = eulerAngles[2] - self.angle_offsets["yaw"]
        
        transformed_orientation_matrix = np.dot(self.inverse_transformation_matrix[:3, :3], euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3])
        transformed_orientation = euler_from_matrix(transformed_orientation_matrix, axes='sxyz')

        return transformed_position, transformed_orientation

def main():
    rclpy.init()
    ar4_robot = AR4_ROBOT()

    # Example usage: Move to a specific pose
    ar4_robot.get_logger().info("Moving to target pose...")
    target_position = [0.3, 0.2, 0.4]
    target_orientation_euler = [np.pi/2, 0.0, 0.0]
    ar4_robot.move_pose(target_position, target_orientation_euler)

    time.sleep(2)

    # Example usage: Move to a specific joint position
    ar4_robot.get_logger().info("Moving to target joint state...")
    target_joint_state = [0.0, -1.0, 1.5, 0.0, 0.5, 0.0]
    ar4_robot.move_to_joint_positions(target_joint_state)

    rclpy.shutdown()

if __name__ == '__main__':
    main()