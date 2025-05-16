#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import sympy as sp
from urdf_parser_py.urdf import URDF
from xacro import process_file
import os
import numpy as np
from functools import partial
import csv
from scipy.spatial.transform import Rotation as R


'''rclpy.init()
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

rclpy.shutdown()'''
symbolic_matrices = {}

fileNameList = ["Joint1.csv", "Joint2.csv", "Joint3.csv", "Joint4.csv", "Joint5.csv", "Joint6.csv", "Joint7.csv", "Joint8.csv",]
for fileName in fileNameList:
    with open(fileName, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header if present
        data = [row for row in reader]

    # Convert to SymPy Matrix
    symbolic_matrix = sp.Matrix([[sp.sympify(cell) for cell in row] for row in data])
    # Store in dictionary with key based on file name (without .csv)
    key = fileName.replace(".csv", "")
    symbolic_matrices[key] = symbolic_matrix

#print(symbolic_matrices["Joint1"])

baseToWrist = sp.eye(4)
for i in range(1, 7):
    key = "Joint" + str(i)
    symbolic_matrix = symbolic_matrices[key]
    baseToWrist = baseToWrist * symbolic_matrix
    #print(i)

#print("Base to Wrist Transformation Matrix:")
#print(baseToWrist)




array = np.loadtxt("verifyFKPoints.csv", delimiter=",", skiprows=1)
moveitPoses = array[:, :7]
joint_positions = array[:, 7:]
poseArray = np.array([])
q1, q2, q3, q4, q5, q6 = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')

for i in range(0, moveitPoses.shape[0]):
    moveitFKResult = moveitPoses[i, :7]
    M_num = baseToWrist.subs({q1: joint_positions[i,0],q2: joint_positions[i,1],q3: joint_positions[i,2],q4: joint_positions[i,3],q5: joint_positions[i,4],q6: joint_positions[i,5],})
    #print(M_num)

    rot_matrix = M_num[:3, :3]
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()
    trans = np.array(M_num[:3, 3]).flatten().T
    row = np.concatenate((trans, quat))
    if poseArray.size == 0:
        poseArray = row.reshape(1, -1)  # Initialize as a 2D array
    else:
        poseArray = np.vstack((poseArray, row))

    myFKResult = row
    #print(moveitFKResult-myFKResult)

meanDifference = np.mean(np.sum(np.abs(poseArray-moveitPoses),axis=1))

print('done')

















