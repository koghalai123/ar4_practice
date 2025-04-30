#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Pose
from moveit_msgs.msg import MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from pymoveit2 import MoveIt2

class WorkEnvelopePlanner(Node):
    def __init__(self):
        super().__init__('work_envelope_planner')
        
        # Initialize MoveIt2 interface
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        
        # Service to check reachability
        self.subscription = self.create_subscription(Pose,
                                                     "/check_reachability",
                                                     self.check_reachability_callback,
                                                     1)
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose",
                                              1)
        
        self.get_logger().info("Work Envelope Planner ready")

    def check_reachability_callback(self, pose):
        try:
            
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "base_link"
            pose_goal.pose = pose

            self.moveit2.plan(pose=pose_goal)
            #self.moveit2.wait_until_executed()
            self.get_logger().info('Received: ')
                
        except Exception as e:
            self.get_logger().error(f"Error in reachability check: {str(e)}")
            

def main(args=None):
    rclpy.init(args=args)
    planner = WorkEnvelopePlanner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()