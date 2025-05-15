#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
# Initialize ROS 2
rclpy.init()

# Create a ROS 2 node
node = rclpy.create_node("fk_example_node")

moveit2 = MoveIt2(
            node=node,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=ReentrantCallbackGroup()
        )

poseArray = np.array([])
n = 1000
random_array = np.random.uniform(-3, 3, (n, 6))
for joint_positions in random_array:
    fk_result = moveit2.compute_fk(joint_positions)
    position = fk_result.pose.position
    position_array = np.array([position.x, position.y, position.z])
    orientation = fk_result.pose.orientation
    orientation_array = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
    pose = np.concatenate((position_array, orientation_array,joint_positions))
    if poseArray.size == 0:
        poseArray = pose.reshape(1, -1)  # Initialize as a 2D array
    else:
        poseArray = np.vstack((poseArray, pose))  # Stack rows
    print("Joint Positions:", joint_positions)

np.savetxt("verifyFKPoints.csv", poseArray, delimiter=",", header="x,y,z,qx,qy,qz,qw,J1,J2,J3,J4,J5,J6", comments="")


# Shutdown ROS 2
rclpy.shutdown()