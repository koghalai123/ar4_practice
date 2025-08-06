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

class CalibrationConvergenceSimulator:
    def __init__(self, n=10, numIters=10, dQMagnitude=0.1, dLMagnitude=0.0,dXMagnitude=0.1):
        self.n=n
        self.m = 6
        self.noiseMagnitude = 0.00
        
                
        self.resetMatrices()
        
        self.dQMagnitude = dQMagnitude
        self.dQ = np.random.uniform(-self.dQMagnitude, self.dQMagnitude, (1, 6))[0]
        
        
        self.LMat = np.ones((1, 6))
        self.dLMagnitude = dLMagnitude
        self.dL = np.random.uniform(-self.dLMagnitude, self.dLMagnitude, (1, 6))[0]
        
        self.XNominal = np.zeros((6))
        self.dXMagnitude = dXMagnitude
        self.dX = np.random.uniform(-self.dXMagnitude, self.dXMagnitude, (1,6))[0]
        self.XActual = self.XNominal + self.dX
        
        self.numIters = numIters
        self.dQMat = np.zeros((self.numIters, 6))
        self.dLMat = np.zeros((self.numIters, 6))
        self.dXMat = np.zeros((self.numIters, 6))
        self.avgAccMat = np.ones((self.numIters,2))
        
        self.symbolic_matrices = self.loadSymbolicTransforms()
        self.baseToWrist = sp.eye(4)
        self.wristToBase = sp.eye(4)
        self.l = sp.symbols('l1:7')
        self.x = sp.symbols('x1:7')
        self.q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        
        # Current iteration index
        self.current_iter = 0
        self.current_sample = 0
        
        self.setup_kinematics()
    
    def resetMatrices(self):
        self.poseArrayActual = np.zeros((self.n, self.m))
        self.poseArrayCommanded = np.zeros((self.n, self.m))
        self.joint_positions_actual = np.zeros((self.n, self.m))
        self.joint_positions_commanded = np.zeros((self.n, self.m))
        self.poseArrayCalibrated = np.zeros((self.n, self.m))
        
        self.numJacobianTrans = np.zeros((0, 18))
        self.numJacobianRot = np.zeros((0, 18))
        
    def loadSymbolicTransforms(self):
        '''symbolic_matrices = {}
        
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
            symbolic_matrices[key] = symbolic_matrix'''
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
        return symbolic_matrices
    
        
    def get_homogeneous_transform(self, xyz, euler_angles, rotation_order='XYZ'):
        rotation = R.from_euler(rotation_order, euler_angles)
        rot_mat = rotation.as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = xyz
        
        return transform
        
    def symbolic_transform_with_ref_frames(self, xyz, euler_angles, rotation_order='XYZ'):
        N = ReferenceFrame('N')  # World frame
        B = N.orientnew('B', 'Body', euler_angles, rotation_order)
        
        R = N.dcm(B)
        
        T = eye(4)
        T[:3, :3] = R.T  # Transpose to get rotation from B to N
        T[:3, 3] = xyz
        
        return T
        
    def setup_kinematics(self):
        self.originToBase = self.symbolic_transform_with_ref_frames(self.x[0:3], [0,0,0], rotation_order='XYZ')
        self.originToBaseActual = self.get_homogeneous_transform(self.XActual[0:3], self.XActual[3:6], rotation_order='XYZ')
        
        '''for i in range(1, 7):
            key = "Joint" + str(i)'''
            
        for i in range(1, 7):
            key = "Joint" + str(i)
            symbolic_matrix = self.symbolic_matrices[key]
            symbolic_matrix = self.symbolic_matrices[key]
            translation_vector = symbolic_matrix[:3, 3]
            norm_symbolic = sp.sqrt(sum(component**2 for component in translation_vector))
            self.LMat[0, i - 1] = norm_symbolic
            symbolic_matrix[:3,3]=symbolic_matrix[:3,3]*self.l[i-1]
            self.baseToWrist = self.baseToWrist * symbolic_matrix
            self.wristToBase = symbolic_matrix.inv() * self.wristToBase
            self.symbolic_matrices[key] = symbolic_matrix
        #baseToWrist = sp.eye(4)
        
        self.originToWrist = self.originToBase*self.baseToWrist
        self.translation_vector = self.originToWrist[:3, 3]
        self.rotation_matrix = self.originToWrist[:3, :3]
        
        '''self.pitch = sp.asin(-self.rotation_matrix[2, 0])
        self.roll = sp.atan2(self.rotation_matrix[2, 1], self.rotation_matrix[2, 2])
        self.yaw = sp.atan2(self.rotation_matrix[1, 0], self.rotation_matrix[0, 0])
        self.euler_angles = sp.Matrix([self.roll, self.pitch, self.yaw])'''
        
        vars = list(self.q) + list(self.l) + list(self.x)
        self.jacobian_translation = self.translation_vector.jacobian(vars)
        
        self.rotation_matrix_flat = self.rotation_matrix.reshape(9, 1)
        self.jacobian_rotation = self.rotation_matrix_flat.jacobian(vars)
        
        self.joint_lengths_nominal = np.ones(6)
        self.joint_lengths_actual = self.joint_lengths_nominal + self.dL.flatten()
    
    def set_current_iteration(self, iteration_index):
        """Set the current iteration index"""
        if iteration_index >= 0 and iteration_index < self.numIters:
            self.current_iter = iteration_index
        else:
            print(f"Warning: Invalid iteration index {iteration_index}. Using 0 instead.")
            self.current_iter = 0
    
    def get_fk_calibration_model(self, joint_positions, joint_lengths, XOffsets):
        l = sp.symbols('l1:7')
        x = sp.symbols('x1:7')
        q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        
        M_num_actual = self.baseToWrist.subs({
            **{q[k]: joint_positions[k] for k in range(6)},
            **{l[k]: joint_lengths[k] for k in range(6)},
            **{x[k]: XOffsets[k] for k in range(6)}
        })   
        '''M_num_inverse = wristToBase.subs({
        **{q[j]: joint_positions[i, j] for j in range(6)},  # Substitute q variables
        **{l[j]: joint_lengths[j] for j in range(6)}               # Substitute l variables
        }) '''
        rot_matrix = M_num_actual[:3, :3]
        trans = np.array(M_num_actual[:3, 3]).flatten().T
        pose = np.concatenate((trans, R.from_matrix(np.array(rot_matrix).astype(np.float64)).as_euler('xyz')))
        
        
        return pose
    
    def generate_measurement_pose(self, robot, pose = None, calibrate=False, frame = "end_effector_link"):
        
        
        if pose is None:
            pose = np.random.uniform(-0.03, 0.03, (1, 6))[0]
        position = pose[:3]
        orientation =  pose[3:6]
        #transformed_position, transformed_orientation = robot.fromMyPreferredFrame(position, orientation, old_reference_frame=frame, new_reference_frame="base_link")
        
        #pos,ori = robot.get_current_pose()
        joint_positions_commanded = robot.get_ik(position=position, orientation_euler=orientation, frame_id=frame)
        
        #joint_lengths = self.joint_lengths_nominal
        #XOffsets = self.XNominal
        
        #
        #pose_fk = self.get_fk_calibration_model(np.array([0,0,0,0,0,0]), joint_lengths, XOffsets)
        #global_position, global_orientation = robot.toMyPreferredFrame(pose_fk[:3], pose_fk[3:6], reference_frame="base_link")
        #fk_result = robot.moveit2.compute_fk(
        #    joint_state=np.array([0,0,0,0,0,0]).tolist(),  # Convert NumPy array to list
        #)
        pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints(joint_positions_commanded, calibrate)
        return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded
        
    def generate_measurement_joints(self, joint_positions_commanded = None, calibrate=False):
        """Generate a single measurement pair (actual and commanded)"""
                

        # Generate actual pose
        #noise = np.random.uniform(-self.noiseMagnitude, self.noiseMagnitude, (1, 6))[0]
        if joint_positions_commanded is None:
            joint_positions_commanded = np.random.uniform(-3, 3, (1, 6))[0]
            
        joint_lengths_commanded = self.joint_lengths_nominal
        XOffsets_commanded = self.XNominal
        pose_commanded = self.get_fk_calibration_model(joint_positions_commanded, joint_lengths_commanded, XOffsets_commanded)
        
        self.joint_positions_commanded[self.current_sample][:] = joint_positions_commanded
        
        
        if calibrate:
            joint_positions_commanded = joint_positions_commanded - np.sum(self.dQMat, axis=0)
        joint_lengths = self.joint_lengths_actual
        XOffsets = self.XActual.flatten()
        joint_positions_actual = joint_positions_commanded + self.dQ
        pose_actual = self.get_fk_calibration_model(joint_positions_actual, joint_lengths, XOffsets)
                
        if calibrate:
            pose_commanded = pose_commanded + np.sum(self.dXMat, axis=0)


        self.poseArrayActual[self.current_sample][:] = pose_actual
        self.joint_positions_actual[self.current_sample][:] = joint_positions_actual        
        self.poseArrayCommanded[self.current_sample][:] = pose_commanded
        
        self.current_sample += 1     
                    
        return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded
    
            
        
    def compute_jacobians(self, joint_angles):
        """Compute Jacobians for all measurements in current iteration"""
        l = sp.symbols('l1:7')
        x = sp.symbols('x1:7')
        q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        vars = list(self.q) + list(self.l) + list(self.x)
        
        num_measurements = 1#len(measurements_actual)
        numJacobianTrans = np.ones((3*num_measurements, len(vars)))
        rotCount = 9
        numJacobianRot = np.ones((rotCount*num_measurements, len(vars)))

        # noise = np.random.uniform(-self.noiseMagnitude, self.noiseMagnitude, (num_measurements, 6))
        # Use only the relevant subset of commanded joint positions
        #joint_positions = self.joint_positions_commanded[:num_measurements] #+ noise
        joint_lengths = self.joint_lengths_nominal + np.sum(self.dLMat, axis=0)
        
        '''for i in range(num_measurements):
            XOffsets = self.XNominal + np.sum(self.dXMat, axis=0)
            partialsTrans = self.jacobian_translation.subs({
                **{q[j]: joint_positions[i, j] for j in range(6)},  # Substitute q variables
                **{l[j]: joint_lengths[j] for j in range(6)},
                **{x[j]: XOffsets[j] for j in range(6)},
            })   
            partialsRot = self.jacobian_rotation.subs({
                **{q[j]: joint_positions[i, j] for j in range(6)},  # Substitute q variables
                **{l[j]: joint_lengths[j] for j in range(6)},
                **{x[j]: XOffsets[j] for j in range(6)},
            })  '''
        XOffsets = self.XNominal + np.sum(self.dXMat, axis=0)
        partialsTrans = self.jacobian_translation.subs({
            **{q[j]: joint_angles[j] for j in range(6)},  # Substitute q variables
            **{l[j]: joint_lengths[j] for j in range(6)},
            **{x[j]: XOffsets[j] for j in range(6)},
        })   
        partialsRot = self.jacobian_rotation.subs({
            **{q[j]: joint_angles[j] for j in range(6)},  # Substitute q variables
            **{l[j]: joint_lengths[j] for j in range(6)},
            **{x[j]: XOffsets[j] for j in range(6)},
        })
        i = 0
        numJacobianTrans[3*i:3*i+3,:] = np.array(partialsTrans).astype(np.float64)
        numJacobianRot[rotCount*i:rotCount*i+rotCount,:] = np.array(partialsRot).astype(np.float64)
        
        
        if self.numJacobianTrans.size == 0:
            self.numJacobianTrans = numJacobianTrans
        else:
            self.numJacobianTrans = np.vstack((self.numJacobianTrans, numJacobianTrans))

        if self.numJacobianRot.size == 0:
            self.numJacobianRot = numJacobianRot
        else:
            self.numJacobianRot = np.vstack((self.numJacobianRot, numJacobianRot))
        
        return numJacobianTrans, numJacobianRot
        
    def compute_differences(self, measurements_actual, measurements_commanded):
        """Compute differences between actual and commanded poses"""
        measurements_actual = np.array(measurements_actual).astype(np.float64)
        measurements_commanded = np.array(measurements_commanded).astype(np.float64)
        
        translationDifferences = measurements_actual[:, :3] - measurements_commanded[:, :3]
        
        rotationalDifferences = []
        for i in range(len(measurements_actual)):
            # Extract Euler angles for the current pose
            euler_actual = measurements_actual[i, 3:6]
            euler_commanded = measurements_commanded[i, 3:6]

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
        
        return translationDifferences, rotationalDifferences
        
    def compute_error_metrics(self, translationDifferences, rotationalDifferences):
        """Compute error metrics from the differences"""
        accuracyError = np.linalg.norm(translationDifferences, axis=1)
        rotationalError = np.linalg.norm(rotationalDifferences, axis=1)
        avgRotationalError = np.mean(rotationalError)
        avgAccuracyError = np.mean(accuracyError)
        avgTransAndRotError = np.array([avgAccuracyError, avgRotationalError])
        
        return avgAccuracyError, avgRotationalError, avgTransAndRotError
        
    def compute_calibration_parameters(self, translationDifferences, rotationalDifferences, numJacobianTrans, numJacobianRot):
        """Compute calibration parameters using least squares"""
        translation_weight = 1.0  # Weight for translational errors
        rotation_weight = 0.0     # Weight for rotational errors
        
        # Scale translational and rotational differences
        scaled_translation_differences = translation_weight * translationDifferences.flatten()
        scaled_rotational_differences = rotation_weight * rotationalDifferences.ravel()

        bMat = np.concatenate((scaled_translation_differences, scaled_rotational_differences))
        AMat = np.vstack((translation_weight * numJacobianTrans, rotation_weight * numJacobianRot))
        
        #bMat = scaled_translation_differences
        #AMat = translation_weight * numJacobianTrans

        
        errorEstimates, residuals, rank, singular_values = np.linalg.lstsq(AMat, bMat, rcond=None)
        
        return errorEstimates
    
    def process_iteration_results(self, measurements_actual, measurements_commanded,numJacobianTrans,numJacobianRot):
        """Process all measurements for the current iteration"""
        j = self.current_iter

            
        # Compute Jacobians
        #numJacobianTrans, numJacobianRot = self.compute_jacobians(measurements_actual, measurements_commanded)
        
        # Compute differences
        translationDifferences, rotationalDifferences = self.compute_differences(measurements_actual, measurements_commanded)
        
        # Compute error metrics
        avgAccuracyError, avgRotationalError, avgTransAndRotError = self.compute_error_metrics(
            translationDifferences, rotationalDifferences)
        self.avgAccMat[j,:] = avgTransAndRotError
        
        # Compute calibration parameters
        errorEstimates = self.compute_calibration_parameters(
            translationDifferences, rotationalDifferences, numJacobianTrans, numJacobianRot)
            
        # Extract parameter updates
        dQEst = errorEstimates[0:6]
        self.dQMat[j, :] = dQEst
        
        dLEst = errorEstimates[6:12]
        self.dLMat[j, :] = dLEst
        
        dXEst = errorEstimates[12:18]
        self.dXMat[j, :] = dXEst
        
        print("Iteration: ", j)
        print("Avg Pose Error: ", avgTransAndRotError)
        print("dLEst: ", np.sum(self.dLMat,axis=0))
        print("dQAct: ", self.dQ)
        print("dQEst: ", np.sum(self.dQMat,axis=0))
        print("dXEst: ", np.sum(self.dXMat,axis=0))
        print("")
        
        self.resetMatrices()
        self.current_sample = 0
        
        return avgTransAndRotError, np.sum(self.dLMat,axis=0), self.dQ, np.sum(self.dQMat,axis=0), np.sum(self.dXMat,axis=0)
    
        
        
    def save_to_csv(self, filename='calibrationData.csv'):
        """Save calibration data to CSV file"""
        arrays = [self.dQ, self.dL, self.dX, np.array([self.noiseMagnitude]), np.array([self.n]), 
                 self.avgAccMat, self.dQMat, self.dLMat, self.dXMat]
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
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


def main(args=None):
    # Create simulator
    simulator = CalibrationConvergenceSimulator(n=8, numIters=10)
    from ar4_robot import AR4_ROBOT
    use_joint_positions = 0
    robot = AR4_ROBOT(use_joint_positions)
    # Process each iteration separately
    for j in range(simulator.numIters):
        print(f"\n--- Starting Iteration {j} ---")
        simulator.set_current_iteration(j)
        
        # Pre-allocate numpy arrays for measurements
        
        valid_measurements = 0

        # Generate individual measurements
        for i in range(simulator.n):
            #poseActual, poseCommanded, joint_positions_actual, joint_positions_commanded = simulator.generate_measurement_joints(calibrate=True)
            simulator.generate_measurement_pose(robot = robot, pose = None, calibrate=True, frame = "end_effector_link")
            
            
            numJacobianTrans, numJacobianRot = simulator.compute_jacobians(simulator.joint_positions_commanded[simulator.current_sample-1])
            
            print(f"Measurement {i}: Generated")

        
        # Process all measurements for this iteration
        results = simulator.process_iteration_results(simulator.poseArrayActual, simulator.poseArrayCommanded,simulator.numJacobianTrans,simulator.numJacobianRot)
    
    # Save results to CSV
    #simulator.save_to_csv()
    
    print('done')

if __name__ == "__main__":
    main()