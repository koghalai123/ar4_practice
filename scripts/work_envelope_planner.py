#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Pose
from moveit_msgs.msg import MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory
from work_envelope_interfaces.srv import CheckReachability

class WorkEnvelopePlanner(Node):
    def __init__(self):
        super().__init__('work_envelope_planner')
        
        # Initialize MoveIt2 interface
        from pymoveit2 import MoveIt2
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        
        # Service to check reachability
        self.srv = self.create_service(
            CheckReachability,
            'check_reachability',
            self.check_reachability_callback
        )
        
        self.get_logger().info("Work Envelope Planner ready")

    def check_reachability_callback(self, request, response):
        try:
            # Get current orientation to maintain it
            current_pose = self.moveit2.get_current_pose()
            if current_pose is None:
                response.success = False
                return response
            
            # Create target pose with requested position and current orientation
            target_pose = Pose()
            target_pose.position = request.position
            target_pose.orientation = current_pose.pose.orientation
            
            # Plan without executing
            trajectory = self.moveit2.plan(
                position=request.position,
                quat_xyzw=[
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w
                ],
                cartesian=False
            )
            
            if trajectory is not None:
                response.success = True
                response.trajectory = trajectory
            else:
                response.success = False
                
        except Exception as e:
            self.get_logger().error(f"Error in reachability check: {str(e)}")
            response.success = False
            
        return response

def main(args=None):
    rclpy.init(args=args)
    planner = WorkEnvelopePlanner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()