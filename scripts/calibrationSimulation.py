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

n = 20
m = 6
noiseMagnitude = 0.005
joint_positions_commanded = np.random.uniform(-3, 3, (n, 6))
poseArrayActual = np.zeros((n, m))
poseArrayCommanded = np.zeros((n, m))
poseArrayCalibrated = np.zeros((n, m))

dQ = np.random.uniform(-0.3, 0.3, (1, 6))
joint_positions_actual = joint_positions_commanded + dQ

noise = np.random.uniform(-noiseMagnitude, noiseMagnitude, (n, 6))
joint_positions = joint_positions_actual + noise
for i in range(0, joint_positions_actual.shape[0]):
    
    q1, q2, q3, q4, q5, q6 = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
    M_num_actual = baseToWrist.subs({q1: joint_positions[i,0],q2: joint_positions[i,1],q3: joint_positions[i,2],q4: joint_positions[i,3],q5: joint_positions[i,4],q6: joint_positions[i,5],})
    rot_matrix = M_num_actual[:3, :3]
    r = R.from_matrix(rot_matrix)
    #quat = r.as_quat()
    euler = r.as_euler('xyz')
    trans = np.array(M_num_actual[:3, 3]).flatten().T
    row = np.concatenate((trans, euler))
    poseArrayActual[i, :] = row
    
numIters = 10
dQMat = np.zeros((numIters, 6))
q1, q2, q3, q4, q5, q6 = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
translation_vector = baseToWrist[:3, 3]
jacobian_translation = translation_vector.jacobian([q1, q2, q3, q4, q5, q6])

for j in range(0, numIters):
    numJacobian = np.ones((3*joint_positions_commanded.shape[0], 6))
    
    noise = np.random.uniform(-noiseMagnitude, noiseMagnitude, (n, 6))
    joint_positions = joint_positions_commanded+np.sum(dQMat,axis=0) + noise
    for i in range(0, joint_positions_commanded.shape[0]):
        
        M_num_commanded = baseToWrist.subs({q1: joint_positions[i,0],q2: joint_positions[i,1],q3: joint_positions[i,2],q4: joint_positions[i,3],q5: joint_positions[i,4],q6: joint_positions[i,5],})
        rot_matrix = M_num_commanded[:3, :3]
        r = R.from_matrix(rot_matrix)
        #quat = r.as_quat()
        euler = r.as_euler('xyz')
        trans = np.array(M_num_commanded[:3, 3]).flatten().T
        row = np.concatenate((trans, euler))
        poseArrayCommanded[i, :] = row
        partials = jacobian_translation.subs({q1: joint_positions[i,0],q2: joint_positions[i,1],q3: joint_positions[i,2],q4: joint_positions[i,3],q5: joint_positions[i,4],q6: joint_positions[i,5],})
        #print(partials)
        numJacobian[3*i:3*i+3,:] = np.array(partials).astype(np.float64)

    translationDifferences = (poseArrayActual-poseArrayCommanded)[:,:3]
    accuracyError= np.linalg.norm(translationDifferences, axis=1)
    avgAccuracyError = np.mean(accuracyError)



    bMat = translationDifferences.flatten()
    AMat = numJacobian
    x, residuals, rank, singular_values = np.linalg.lstsq(AMat, bMat, rcond=None)

    
    dQEst = x
    dQMat[j, :] = dQEst
    
    '''joint_positions = joint_positions_commanded+x
    for i in range(0, joint_positions_commanded.shape[0]):
        
        M_num_calibrated = baseToWrist.subs({q1: joint_positions[i,0],q2: joint_positions[i,1],q3: joint_positions[i,2],q4: joint_positions[i,3],q5: joint_positions[i,4],q6: joint_positions[i,5],})
        rot_matrix = M_num_calibrated[:3, :3]
        r = R.from_matrix(rot_matrix)
        #quat = r.as_quat()
        euler = r.as_euler('xyz')
        trans = np.array(M_num_calibrated[:3, 3]).flatten().T
        row = np.concatenate((trans, euler))
        poseArrayCalibrated[i, :] = row
        partials = jacobian_translation.subs({q1: joint_positions[i,0],q2: joint_positions[i,1],q3: joint_positions[i,2],q4: joint_positions[i,3],q5: joint_positions[i,4],q6: joint_positions[i,5],})
        #print(partials)
        numJacobian[3*i:3*i+3,:] = np.array(partials).astype(np.float64)

    translationDifferencesCalibrated = (poseArrayActual-poseArrayCalibrated)[:,:3]
    accuracyErrorCalibrated = np.linalg.norm(translationDifferencesCalibrated, axis=1)
    avgAccuracyErrorCalibrated = np.mean(accuracyErrorCalibrated)'''
    print("Iteration: ", j)
    print("Avg Accuracy Error: ", avgAccuracyError)
    print("dQSum: ", np.sum(dQMat,axis=0))
    print("dQEst: ", dQEst)

print('done')