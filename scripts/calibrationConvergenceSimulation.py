#!/usr/bin/env python3
"""
Robot Calibration Convergence Simulation

This module provides a simulation framework for robot calibration with support for both
standard end-effector position measurements and camera-based target displacement measurements.

Camera Mode Usage:
    # Initialize with camera mode enabled
    sim = CalibrationSimulator(
        n=20,                    # Number of measurements per iteration
        numIters=10,            # Number of calibration iterations
        camera_mode=True,       # Enable camera-based measurements
        camera_transform=[0.05, 0.0, 0.02, 0.0, 0.0, 0.0]  # Known camera mount transform
    )
    
    # Generate camera measurements (joint positions and measured displacements to targets)
    joint_positions, target_displacements = sim.generate_measurement_joints_camera(target_positions_world)
    
    # Camera mode changes only the measurement method - same 18 parameters estimated:
    # - 6 joint encoder errors (dQ)
    # - 6 link length errors (dL) 
    # - 6 base frame errors (dX)
    # The camera transformation is known/fixed (no additional error parameters)
    
Standard Mode Usage:
    # Initialize with standard mode (default)
    sim = CalibrationSimulator(
        n=20,
        numIters=10,
        camera_mode=False       # Standard end-effector measurements
    )
"""

import rclpy
from rclpy.node import Node
#from pymoveit2 import MoveIt2
from ar4_robot_py import AR4Robot
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

class CalibrationSimulator:
    def __init__(self, n=10, numIters=10, dQMagnitude=0.1, dLMagnitude=0.0, dXMagnitude=0.1, 
                 camera_mode=False, camera_transform=None):
        self.n=n
        self.m = 6
        self.noiseMagnitude = 0.00
        
        # Camera mode parameters
        self.camera_mode = camera_mode
        # Fixed camera transformation from end-effector to camera (no error parameters)
        if camera_mode and camera_transform is None:
            # Default: camera 5cm forward, 2cm up from end-effector
            self.camera_transform = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]  # [px, py, pz, rx, ry, rz]
        else:
            self.camera_transform = camera_transform or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
                
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
        
        # No camera error parameters - only robot kinematic errors
        self.numParameters = 18  # Always 6 joint + 6 length + 6 base
        
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
        
        # Jacobian matrices always have 18 parameters (no camera error parameters)
        self.numJacobianTrans = np.zeros((0, 18))
        self.numJacobianRot = np.zeros((0, 18))
        
        # For camera mode, store target displacement measurements
        if self.camera_mode:
            self.target_displacements_measured = np.zeros((self.n, 3))
            self.target_displacements_expected = np.zeros((self.n, 3))
        
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
        
        # Set up transformation chain
        if self.camera_mode:
            # Camera mode: origin -> base -> wrist -> camera
            # Fixed camera transformation (no symbolic parameters)
            px, py, pz = self.camera_transform[:3]
            rx, ry, rz = self.camera_transform[3:]
            
            # Create fixed camera transform with numerical values
            self.wristToCamera = self.symbolic_transform_with_ref_frames(
                [px, py, pz], [rx, ry, rz], rotation_order='XYZ')
            
            # Complete transformation chain to camera
            self.originToCamera = self.originToBase * self.baseToWrist * self.wristToCamera
            self.translation_vector = self.originToCamera[:3, 3]
            self.rotation_matrix = self.originToCamera[:3, :3]
        else:
            # Standard mode: origin -> base -> wrist
            self.originToWrist = self.originToBase*self.baseToWrist
            self.translation_vector = self.originToWrist[:3, 3]
            self.rotation_matrix = self.originToWrist[:3, :3]
        
        # Variable list is always the same (no camera parameters)
        vars = list(self.q) + list(self.l) + list(self.x)
        
        '''self.pitch = sp.asin(-self.rotation_matrix[2, 0])
        self.roll = sp.atan2(self.rotation_matrix[2, 1], self.rotation_matrix[2, 2])
        self.yaw = sp.atan2(self.rotation_matrix[1, 0], self.rotation_matrix[0, 0])
        self.euler_angles = sp.Matrix([self.roll, self.pitch, self.yaw])'''
        
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
    
    def get_camera_position_actual(self, joint_positions):
        """Get actual camera position (with robot errors) for camera mode"""
        if not self.camera_mode:
            raise ValueError("Camera mode must be enabled")
            
        # Use actual joint lengths and base offsets (with errors)
        joint_lengths = self.joint_lengths_actual
        XOffsets = self.XActual.flatten()
        
        l = sp.symbols('l1:7')
        x = sp.symbols('x1:7')
        q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        
        # Camera transformation includes the fixed camera mount
        M_camera = self.originToCamera.subs({
            **{q[k]: joint_positions[k] for k in range(6)},
            **{l[k]: joint_lengths[k] for k in range(6)},
            **{x[k]: XOffsets[k] for k in range(6)}
        })
        
        rot_matrix = M_camera[:3, :3]
        trans = np.array(M_camera[:3, 3]).flatten().T
        pose = np.concatenate((trans, R.from_matrix(np.array(rot_matrix).astype(np.float64)).as_euler('xyz')))
        
        return pose
    
    def get_camera_position_commanded(self, joint_positions):
        """Get commanded camera position (without robot errors) for camera mode"""
        if not self.camera_mode:
            raise ValueError("Camera mode must be enabled")
            
        # Use nominal joint lengths and base offsets (no errors)
        joint_lengths = self.joint_lengths_nominal
        XOffsets = self.XNominal
        
        l = sp.symbols('l1:7')
        x = sp.symbols('x1:7')
        q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        
        # Camera transformation includes the fixed camera mount
        M_camera = self.originToCamera.subs({
            **{q[k]: joint_positions[k] for k in range(6)},
            **{l[k]: joint_lengths[k] for k in range(6)},
            **{x[k]: XOffsets[k] for k in range(6)}
        })
        
        rot_matrix = M_camera[:3, :3]
        trans = np.array(M_camera[:3, 3]).flatten().T
        pose = np.concatenate((trans, R.from_matrix(np.array(rot_matrix).astype(np.float64)).as_euler('xyz')))
        
        return pose
    
    def generate_measurement_pose(self, robot, pose = None, calibrate=False, frame = "end_effector_link"):
        
        
        if pose is None:
            pose = np.random.uniform(-0.03, 0.03, (1, 6))[0]
        position = pose[:3]
        orientation =  pose[3:6]
        #transformed_position, transformed_orientation = robot.fromMyPreferredFrame(position, orientation, old_reference_frame=frame, new_reference_frame="base_link")
        
        #pos,ori = robot.get_current_pose()
        joint_positions_commanded = robot.get_ik(position=position, euler_angles=orientation, frame_id=frame)
        
        #joint_lengths = self.joint_lengths_nominal
        #XOffsets = self.XNominal
        
        #
        #pose_fk = self.get_fk_calibration_model(np.array([0,0,0,0,0,0]), joint_lengths, XOffsets)
        #global_position, global_orientation = robot.toMyPreferredFrame(pose_fk[:3], pose_fk[3:6], reference_frame="base_link")
        #fk_result = robot.moveit2.compute_fk(
        #    joint_state=np.array([0,0,0,0,0,0]).tolist(),  # Convert NumPy array to list
        #)
        if self.camera_mode:
            target_positions_world=None
            pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints_camera(target_positions_world, joint_positions_commanded, calibrate)
        else:
            pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints(joint_positions_commanded, calibrate)

        return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded

    def generate_measurement_joints_camera(self, target_positions_world=None, joint_positions_commanded=None, calibrate=True):
        """Generate camera-based measurements using displacement vectors to known targets
        
        Args:
            target_positions_world: Known target positions in world coordinates (n, 3)
            
        Returns:
            joint_positions_commanded: Commanded joint positions (n, 6)
            target_displacements: Measured displacement vectors from camera to targets in camera frame (n, 3)
        """
        if not self.camera_mode:
            raise ValueError("Camera mode must be enabled to use this method")
            
        if target_positions_world is None:
            # Generate random target positions in world frame
            target_positions_world = np.array([0.3,0,0])
            
        # Generate random joint positions for measurements
        #joint_positions_commanded = np.random.uniform(-2.5, 2.5, (self.n, 6))
        
        # Store commanded joint positions
        self.joint_positions_commanded = joint_positions_commanded
        
        # Calculate actual joint positions with errors
        self.joint_positions_actual = joint_positions_commanded + self.dQ
        
        # For each measurement, calculate the displacement from camera to target
        target_displacements = np.zeros((self.n, 3))
        
        # Get actual camera position in world frame (with robot errors)
        camera_pose_actual = self.get_camera_position_actual(self.joint_positions_actual)
        camera_position_actual = camera_pose_actual[:3]
        camera_rotation_actual = R.from_euler('xyz', camera_pose_actual[3:6]).as_matrix()
        
        # Calculate displacement vector from camera to target in world frame
        displacement_world = target_positions_world - camera_position_actual
        
        # Transform displacement to camera frame
        displacement_camera = camera_rotation_actual.T @ displacement_world
        
        # Add measurement noise
        #noise = np.random.uniform(-self.noiseMagnitude, self.noiseMagnitude, 3)
        target_displacements = displacement_camera #+ noise
        
        # Store for processing
        self.target_displacements_measured = target_displacements
        
        # Calculate expected displacement using commanded/nominal parameters
        camera_pose_commanded = self.get_camera_position_commanded(joint_positions_commanded)
        camera_position_commanded = camera_pose_commanded[:3]
        camera_rotation_commanded = R.from_euler('xyz', camera_pose_commanded[3:6]).as_matrix()
        
        displacement_world_expected = target_positions_world - camera_position_commanded
        displacement_camera_expected = camera_rotation_commanded.T @ displacement_world_expected
        self.target_displacements_expected = displacement_camera_expected
            
        return joint_positions_commanded, target_displacements
    
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
        
        # Variable list is always the same (no camera parameters to estimate)
        vars = list(self.q) + list(self.l) + list(self.x)
        
        num_measurements = 1
        numJacobianTrans = np.ones((3*num_measurements, len(vars)))
        rotCount = 9
        numJacobianRot = np.ones((rotCount*num_measurements, len(vars)))

        joint_lengths = self.joint_lengths_nominal + np.sum(self.dLMat, axis=0)
        XOffsets = self.XNominal + np.sum(self.dXMat, axis=0)
        
        # Prepare substitution dictionary (no camera parameters)
        subs_dict = {
            **{q[j]: joint_angles[j] for j in range(6)},  # Substitute q variables
            **{l[j]: joint_lengths[j] for j in range(6)},
            **{x[j]: XOffsets[j] for j in range(6)},
        }
        
        partialsTrans = self.jacobian_translation.subs(subs_dict)
        partialsRot = self.jacobian_rotation.subs(subs_dict)
        
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
        """Compute differences between actual and commanded poses/displacements"""
        if self.camera_mode:
            # For camera mode, use displacement differences directly
            translationDifferences = self.target_displacements_measured - self.target_displacements_expected
            # No rotational differences for camera displacement measurements
            rotationalDifferences = np.zeros((self.n, 9))
        else:
            # Standard mode: pose differences
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
            
        # Extract parameter updates (always 18 parameters)
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
        if self.camera_mode:
            print("Camera mode: Using displacement measurements")
        print("")
        
        self.resetMatrices()
        self.current_sample = 0
        
        return avgTransAndRotError, np.sum(self.dLMat,axis=0), self.dQ, np.sum(self.dQMat,axis=0), np.sum(self.dXMat,axis=0)
    
        
        
    def save_to_csv(self, filename='calibrationData.csv'):
        """Save calibration data to CSV file"""
        arrays = [self.dQ, self.dL, self.dX, np.array([self.noiseMagnitude]), np.array([self.n]), 
                 self.avgAccMat, self.dQMat, self.dLMat, self.dXMat]
        prefixes = ['dQ', 'dL', 'dX', 'noiseMagnitude', 'n', 'avgAccMat', 'dQMat', 'dLMat', 'dXMat']
        
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
    simulator = CalibrationSimulator(n=8, numIters=10,camera_mode=True,dQMagnitude=0.1, dLMagnitude=0.0, dXMagnitude=0.1)
    
    robot = AR4Robot()
    robot.disable_logging()
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