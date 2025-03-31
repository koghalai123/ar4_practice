#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2

class JointCommander(Node):
    def __init__(self):
        super().__init__("joint_commander")
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        
        self.moveit2.max_velocity = 0.75
        self.moveit2.max_acceleration = 0.5
        
        self.get_logger().info("Joint commander initialized. Ready to accept commands.")

    def run(self):
        while rclpy.ok():
            try:
                input_str = input("\nEnter joint angles (radians, comma separated): ")
                joint_positions = [float(x.strip()) for x in input_str.split(',')]
                self.moveit2.move_to_configuration(joint_positions)
                self.moveit2.wait_until_executed()
            except Exception as e:
                self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    commander = JointCommander()
    commander.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()