#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, eye
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import init_vprinting
from urdf_parser_py.urdf import URDF
from xacro import process_file
import os
import numpy as np
from functools import partial
import csv
from scipy.spatial.transform import Rotation as R
import pandas as pd

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
def get_homogeneous_transform(xyz, euler_angles, rotation_order='XYZ'):
    rotation = R.from_euler(rotation_order, euler_angles)
    rot_mat = rotation.as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = xyz
    
    return transform

n = 7
m = 6
noiseMagnitude = 0.00

joint_positions_commanded = np.random.uniform(-3, 3, (n, 6))
poseArrayActual = np.zeros((n, m))
poseArrayCommanded = np.zeros((n, m))
poseArrayCalibrated = np.zeros((n, m))

dQMagnitude = 0.15
dQ = np.random.uniform(-dQMagnitude, dQMagnitude, (1, 6))
joint_positions_actual = joint_positions_commanded + dQ

LMat = np.ones((1, 6))
dLMagnitude = 0.0
dL = np.random.uniform(-dLMagnitude, dLMagnitude, (1, 6))

XNominal = np.zeros((6))
dXMagnitude = 0.0
dX = np.random.uniform(-dXMagnitude, dXMagnitude, (1,6))
XActual = XNominal + dX
originToBaseActual = get_homogeneous_transform(XActual[0,0:3], XActual[0,3:6], rotation_order='XYZ')

baseToWrist = sp.eye(4)
wristToBase = sp.eye(4)
l = sp.symbols('l1:7')
x = sp.symbols('x1:7')

def symbolic_transform_with_ref_frames(xyz, euler_angles, rotation_order='XYZ'):
 
    N = ReferenceFrame('N')  # World frame
    B = N.orientnew('B', 'Body', euler_angles, rotation_order)
    
    R = N.dcm(B)
    
    T = eye(4)
    T[:3, :3] = R.T  # Transpose to get rotation from B to N
    T[:3, 3] = xyz
    
    return T

#originToBase = symbolic_transform_with_ref_frames(x[0:3], x[3:6], rotation_order='XYZ')
originToBase = symbolic_transform_with_ref_frames(x[0:3], [0,0,0], rotation_order='XYZ')
for i in range(1, 7):
    key = "Joint" + str(i)
    symbolic_matrix = symbolic_matrices[key]
    translation_vector = symbolic_matrix[:3, 3]
    norm_symbolic = sp.sqrt(sum(component**2 for component in translation_vector))
    LMat[0, i - 1] = norm_symbolic
    symbolic_matrix[:3,3]=symbolic_matrix[:3,3]*l[i-1]
    baseToWrist = baseToWrist * symbolic_matrix
    wristToBase = symbolic_matrix.inv() * wristToBase
    symbolic_matrices[key] = symbolic_matrix
    #print(i)

numIters = 8
dQMat = np.zeros((numIters, 6))
dLMat = np.zeros((numIters, 6))
dXMat = np.zeros((numIters, 6))
avgAccMat = np.ones((numIters,1))


q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')

originToWrist = originToBase*baseToWrist
translation_vector = originToWrist[:3, 3]
rotation_matrix = originToWrist[:3, :3]


pitch = sp.asin(-rotation_matrix[2, 0])  # pitch = arcsin(-r31)
roll = sp.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # roll = atan2(r32, r33)
yaw = sp.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # yaw = atan2(r21, r11)
euler_angles = sp.Matrix([roll, pitch, yaw])


vars = list(q) + list(l) + list(x)
jacobian_translation = translation_vector.jacobian(vars)
jacobian_rotation = euler_angles.jacobian(vars)


rotation_matrix_flat = rotation_matrix.reshape(9, 1)
jacobian_rotation = rotation_matrix_flat.jacobian(vars)

joint_lengths_nominal = LMat.flatten()
joint_lengths_actual = joint_lengths_nominal + dL.flatten()


for j in range(0, numIters):
    l = sp.symbols('l1:7')
    x = sp.symbols('x1:7')
    q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')

    noise = np.random.uniform(-noiseMagnitude, noiseMagnitude, (n, 6))
    joint_positions = joint_positions_actual + noise - np.sum(dQMat,axis=0)
    joint_lengths = joint_lengths_actual
    XOffsets = XActual.flatten()
    for i in range(0, joint_positions_actual.shape[0]):

        M_num_actual = originToWrist.subs({
        **{q[j]: joint_positions[i, j] for j in range(6)},
        **{l[j]: joint_lengths[j] for j in range(6)},
        **{x[j]: XOffsets[j] for j in range(6)}
        })   
        '''M_num_inverse = wristToBase.subs({
        **{q[j]: joint_positions[i, j] for j in range(6)},  # Substitute q variables
        **{l[j]: joint_lengths[j] for j in range(6)}               # Substitute l variables
        }) '''
        rot_matrix = M_num_actual[:3, :3]
        trans = np.array(M_num_actual[:3, 3]).flatten().T
        #row = np.concatenate((trans, rot_matrix.flatten()))
        row = np.concatenate((trans, R.from_matrix(np.array(rot_matrix).astype(np.float64)).as_euler('xyz')))
        poseArrayActual[i, :] = row

    numJacobianTrans = np.ones((3*joint_positions_commanded.shape[0], len(vars)))
    
    rotCount = 9
    numJacobianRot = np.ones((rotCount*joint_positions_commanded.shape[0], len(vars)))




    noise = np.random.uniform(-noiseMagnitude, noiseMagnitude, (n, 6))
    joint_positions = joint_positions_commanded + noise
    #joint_lengths = joint_lengths_nominal + np.sum(dLMat,axis=0)
    joint_lengths = joint_lengths_nominal + np.sum(dLMat,axis=0)

    for i in range(0, joint_positions_commanded.shape[0]):
        XOffsets = XNominal + np.sum(dXMat,axis=0)
        M_num_commanded = originToWrist.subs({
            **{q[j]: joint_positions[i, j] for j in range(6)},  # Substitute q variables
            **{l[j]: joint_lengths[j] for j in range(6)},
            **{x[j]: XOffsets[j] for j in range(6)},
        })   
        rot_matrix = M_num_commanded[:3, :3]
        trans = np.array(M_num_commanded[:3, 3]).flatten().T
        row = np.concatenate((trans, R.from_matrix(np.array(rot_matrix).astype(np.float64)).as_euler('xyz')))
        poseArrayCommanded[i, :] = row
        partialsTrans = jacobian_translation.subs({
            **{q[j]: joint_positions[i, j] for j in range(6)},  # Substitute q variables
            **{l[j]: joint_lengths[j] for j in range(6)},
            **{x[j]: XOffsets[j] for j in range(6)},
        })   
        partialsRot = jacobian_rotation.subs({
            **{q[j]: joint_positions[i, j] for j in range(6)},  # Substitute q variables
            **{l[j]: joint_lengths[j] for j in range(6)},
            **{x[j]: XOffsets[j] for j in range(6)},
        })  
        #print(partials)
        numJacobianTrans[3*i:3*i+3,:] = np.array(partialsTrans).astype(np.float64)
        numJacobianRot[rotCount*i:rotCount*i+rotCount,:] = np.array(partialsRot).astype(np.float64)

    translationDifferences = (poseArrayActual-poseArrayCommanded)[:,:3]
    rotationalDifferences = (poseArrayActual-poseArrayCommanded)[:,3:6]
    
    rotationalDifferences = []
    for i in range(poseArrayActual.shape[0]):
        # Extract Euler angles for the current pose
        euler_actual = poseArrayActual[i, 3:6]
        euler_commanded = poseArrayCommanded[i, 3:6]

        # Convert Euler angles to rotation matrices
        R_actual = R.from_euler('xyz', euler_actual).as_matrix()
        R_commanded = R.from_euler('xyz', euler_commanded).as_matrix()

        # Compute the difference between the rotation matrices
        R_diff = R_actual - R_commanded

        # Flatten the difference matrix
        R_diff_flat = R_diff.flatten()
        rotationalDifferences.append(R_diff_flat)

    # Convert rotationalDifferences to a NumPy array
    rotationalDifferences = np.array(rotationalDifferences).reshape(-1, 9)
    
    
    accuracyError= np.linalg.norm(translationDifferences, axis=1)
    avgAccuracyError = np.mean(accuracyError)
    avgAccMat[j,0] = avgAccuracyError

    '''#For only measuring translation differences
    bMat = translationDifferences.flatten()
    AMat = numJacobianTrans
    
    #For measuring only rotational differences
    bMat = rotationalDifferences.ravel()
    AMat = numJacobianRot'''
    
    translation_weight = 1.0  # Weight for translational errors
    rotation_weight = 1.0     # Weight for rotational errors
    
    # Scale translational and rotational differences
    scaled_translation_differences = translation_weight * translationDifferences.flatten()
    scaled_rotational_differences = rotation_weight * rotationalDifferences.ravel()

    # Combine scaled errors into a single error vector
    bMat = np.concatenate((scaled_translation_differences, scaled_rotational_differences))

    # Combine Jacobians into a single Jacobian matrix
    AMat = np.vstack((translation_weight * numJacobianTrans, rotation_weight * numJacobianRot))
    
    errorEstimates, residuals, rank, singular_values = np.linalg.lstsq(AMat, bMat, rcond=None)

    dQEst = errorEstimates[0:6]
    dQMat[j, :] = dQEst
    dLEst = errorEstimates[6:12]
    dLMat[j, :] = dLEst
    dXEst = errorEstimates[12:18]
    dXMat[j, :] = dXEst
    print("Iteration: ", j)
    print("Avg Accuracy Error: ", avgAccuracyError)
    print("dLEst: ", np.sum(dLMat,axis=0))
    print("dQAct: ", dQ)
    print("dQEst: ", np.sum(dQMat,axis=0))
    print("dXEst: ", np.sum(dXMat,axis=0))
    print("")




arrays = [dQ, dL, dX, np.array([noiseMagnitude]), np.array([n]), avgAccMat, dQMat, dLMat, dXMat]
max_len = max(arr.shape[0] for arr in arrays)

padded = []
for arr in arrays:
    arr = np.atleast_2d(arr).astype(float)  # Ensure float dtype for NaN padding
    # If arr is shape (1, N), transpose to (N, 1) for 1D arrays
    if arr.shape[0] == 1 and arr.shape[1] != 1 and arr.shape[1] < max_len:
        arr = arr.T
    pad_width = ((0, max_len - arr.shape[0]), (0, 0))
    arr_padded = np.pad(arr, pad_width, constant_values=np.nan)
    padded.append(arr_padded)

combined = np.hstack(padded)

columns = []
prefixes = ['dQ', 'dL', 'dX', 'noiseMagnitude', 'n', 'avgAccMat', 'dQMat', 'dLMat', 'dXMat']

for arr, prefix in zip(arrays, prefixes):
    arr = np.atleast_2d(arr)
    n_cols = arr.shape[1]
    # For scalars or 1D arrays, just use the prefix
    if n_cols == 1:
        columns.append(prefix)
    else:
        columns.extend([f"{prefix}{i+1}" for i in range(n_cols)])

df = pd.DataFrame(combined, columns=columns)
df.to_csv('calibrationData.csv', index=False)



print('done')