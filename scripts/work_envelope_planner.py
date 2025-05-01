#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Pose
from moveit_msgs.msg import MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from pymoveit2 import MoveIt2
import debugpy
import asyncio
from rclpy.callback_groups import ReentrantCallbackGroup
from tf_transformations import quaternion_from_euler, euler_from_quaternion  # Import for Euler/Quaternion conversion
from std_msgs.msg import Bool  # Add this import at the top

#debugpy.listen(("localhost", 5678))
#debugpy.wait_for_client() 

class MoveItCommander(Node):
    def __init__(self):
        super().__init__("moveit_commander")
        # Use ReentrantCallbackGroup for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize MoveIt2 with the callback group
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=self.callback_group
        )
        
        # Create subscription with the callback group
        self.reachability_subscription = self.create_subscription(
            Pose,
            "/check_reachability",
            self.check_reachability_callback,
            1,
            callback_group=self.callback_group
        )

        self.reachability_publisher = self.create_publisher(Bool, "/reachability_result",1)

        self.current_pose = self.create_publisher(PoseStamped, "/get_current_pose",1)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.get_current_pose,callback_group=self.callback_group)


        self.moveit2.max_velocity = 0.75
        self.moveit2.max_acceleration = 0.5
        self.use_joint_positions = 0
        self.get_logger().info("Work Envelope Planner ready")
    def check_reachability_callback(self, pose):
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "base_link"
            pose_goal.pose = pose
            
#await asyncio.wait_for(self.moveit2.compute_ik_async(position=pose.position,quat_xyzw=pose.orientation),timeout=0.5)

            ik_result = self.moveit2.compute_ik(position=pose.position,quat_xyzw=pose.orientation)
            msg = Bool()
            if ik_result is None:
                msg.data = False
            else:
                msg.data = True
            self.reachability_publisher.publish(msg)

    def get_current_pose(self):
        """Get the current end effector pose and print it"""
        current_pose = self.moveit2.compute_fk()
        self.current_pose.publish(current_pose)
            

def main(args=None):
    rclpy.init(args=args)
    planner = MoveItCommander()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()