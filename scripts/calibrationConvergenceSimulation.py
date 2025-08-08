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
                 camera_mode=False, camera_transform=None, target_orientation=None):
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
        
        # Target orientation in world frame (fixed orientation that camera must measure)
        if target_orientation is None:
            # Default: target rotated 30 degrees about each axis
            self.target_orientation_world = np.array([np.pi/6, np.pi/4, np.pi/3])  # [roll, pitch, yaw] in radians
        else:
            self.target_orientation_world = np.array(target_orientation)
        
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
        
        # No additional error parameters - only robot kinematic errors
        self.numParameters = 18  # 6 joint + 6 length + 6 base
        
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
        self.targetArrayActual = np.zeros((self.n, self.m))
        self.targetArrayCommanded = np.zeros((self.n, self.m))
        self.joint_positions_actual = np.zeros((self.n, self.m))
        self.joint_positions_commanded = np.zeros((self.n, self.m))
        self.poseArrayCalibrated = np.zeros((self.n, self.m))
        
        # Jacobian matrices have 18 parameters (robot parameters only)
        self.numJacobianTrans = np.zeros((0, 18))
        self.numJacobianRot = np.zeros((0, 18))
        
        # For camera mode, store both displacement and full transform measurements
        if self.camera_mode:
            self.target_displacements_measured = np.zeros((self.n, 3))
            self.target_displacements_expected = np.zeros((self.n, 3))
            # New: store full 6DOF transformations from camera to target
            self.target_transforms_measured = np.zeros((self.n, 6))
            self.target_transforms_expected = np.zeros((self.n, 6))
        
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
        # Set up transformation chain
        if self.camera_mode:
            # Camera mode: origin -> base -> wrist -> camera -> target (imagined end effector)
            # Fixed camera transformation (no symbolic parameters)
            px, py, pz = self.camera_transform[:3]
            rx, ry, rz = self.camera_transform[3:]
            
            # Create fixed camera transform with numerical values
            self.wristToCamera = self.symbolic_transform_with_ref_frames(
                [px, py, pz], [rx, ry, rz], rotation_order='XYZ')
            
            # Complete transformation chain to camera
            self.originToCamera = self.originToBase * self.baseToWrist * self.wristToCamera
            
            # Create symbolic camera-to-target transformation
            # These symbolic variables will be substituted with measured values
            self.measured_target_position = sp.symbols('tx ty tz')  # Target position in camera frame
            self.measured_target_orientation = sp.symbols('rx_t ry_t rz_t')  # Target orientation in camera frame
            
            # Symbolic camera-to-target transformation based on measurements
            self.cameraToTarget = self.symbolic_transform_with_ref_frames(
                list(self.measured_target_position), 
                list(self.measured_target_orientation), 
                rotation_order='XYZ'
            )
            
            # Complete transformation chain: origin -> camera -> target (imagined end effector)
            self.originToTarget = self.originToCamera * self.cameraToTarget
            
            # Use target pose as the endpoint for Jacobian computation
            # This represents the "imagined end effector" at the target location
            self.translation_vector = self.originToTarget[:3, 3]
            self.rotation_matrix = self.originToTarget[:3, :3]
        else:
            # Standard mode: origin -> base -> wrist
            self.translation_vector = self.originToWrist[:3, 3]
            self.rotation_matrix = self.originToWrist[:3, :3]
        
        # Variable list includes only robot parameters (no target error parameters)
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
    
    def get_fk_calibration_model(self, joint_positions, joint_lengths, XOffsets, camera_to_target=None):
        l = sp.symbols('l1:7')
        x = sp.symbols('x1:7')
        q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        
        if self.camera_mode:
            # Camera mode: use origin to target transformation (imagined end effector)
            subs_dict = {
                **{q[k]: joint_positions[k] for k in range(6)},
                **{l[k]: joint_lengths[k] for k in range(6)},
                **{x[k]: XOffsets[k] for k in range(6)}
            }
            
            # Add camera-to-target pose substitutions if provided
            if camera_to_target is not None:
                subs_dict.update({
                    self.measured_target_position[0]: camera_to_target[0],  # tx
                    self.measured_target_position[1]: camera_to_target[1],  # ty
                    self.measured_target_position[2]: camera_to_target[2],  # tz
                    self.measured_target_orientation[0]: camera_to_target[3],  # rx_t
                    self.measured_target_orientation[1]: camera_to_target[4],  # ry_t
                    self.measured_target_orientation[2]: camera_to_target[5],  # rz_t
                })
            else:
                # Use zero camera-to-target transform (camera pose)
                subs_dict.update({
                    self.measured_target_position[0]: 0.0,
                    self.measured_target_position[1]: 0.0,
                    self.measured_target_position[2]: 0.0,
                    self.measured_target_orientation[0]: 0.0,
                    self.measured_target_orientation[1]: 0.0,
                    self.measured_target_orientation[2]: 0.0,
                })
            
            M_num_actual = self.originToTarget.subs(subs_dict)
        else:
            # Standard mode: use origin to wrist transformation
            M_num_actual = self.originToWrist.subs({
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
            pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints_camera(joint_positions_commanded=joint_positions_commanded, calibrate=calibrate)
        else:
            pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints(joint_positions_commanded=joint_positions_commanded, calibrate=calibrate)

        return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded

    def generate_measurement_joints_camera(self, joint_positions_commanded=None, calibrate=True):
        """Generate camera-based measurements using full homogeneous transformations to known targets
        
        Args:
            joint_positions_commanded: Commanded joint positions for the measurement
            calibrate: Whether to apply calibration corrections
            
        Returns:
            pose_actual: Actual poses (6DOF: position + orientation)
            pose_commanded: Commanded poses (6DOF: position + orientation)
            joint_positions_actual: Actual joint positions
            joint_positions_commanded: Commanded joint positions
        """
        if not self.camera_mode:
            raise ValueError("Camera mode must be enabled to use this method")
        
        # Generate standard joint measurements first
        pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints(joint_positions_commanded=joint_positions_commanded, calibrate=calibrate)
        
        # Set up target position in world frame
        if self.target_positions_world is None:
            # Generate default target position in world frame
            target_positions_world = np.array([0.0, -0.3, 0.0])
        else:
            target_positions_world = self.target_positions_world
        target_orientation_world = self.target_orientation_world
        # For camera measurements, we calculate the transformation from camera to target
        # This gives us both position and orientation of the target relative to the camera
        
        # === ACTUAL MEASUREMENT (with robot errors) ===
        # Get actual camera pose in world frame (with robot errors)
        joint_lengths = self.joint_lengths_actual
        XOffsets = self.XActual.flatten()
        camera_pose_actual = self.get_fk_calibration_model(
            joint_positions=joint_positions_actual, 
            joint_lengths=joint_lengths, 
            XOffsets=XOffsets
        )
        camera_position_actual = camera_pose_actual[:3]
        camera_rotation_actual = R.from_euler('xyz', camera_pose_actual[3:6]).as_matrix()
        
        # Create homogeneous transformation from world to camera (actual)
        T_world_to_camera_actual = np.eye(4)
        T_world_to_camera_actual[:3, :3] = camera_rotation_actual.T  # Inverse rotation
        T_world_to_camera_actual[:3, 3] = -camera_rotation_actual.T @ camera_position_actual
        
        # Target position in homogeneous coordinates
        target_homogeneous = np.append(target_positions_world, 1)
        
        # Transform target from world to camera frame (actual measurement)
        target_in_camera_actual = T_world_to_camera_actual @ target_homogeneous
        target_position_camera_actual = target_in_camera_actual[:3]
        
        # Calculate target orientation relative to camera frame
        # Target's rotation matrix in world frame
        R_target_world = R.from_euler('xyz', self.target_orientation_world).as_matrix()
        
        # Transform target orientation from world to camera frame
        # R_target_camera = R_camera_world^T @ R_target_world
        R_target_camera_actual = camera_rotation_actual.T @ R_target_world
        
        # Convert rotation matrix back to Euler angles
        target_orientation_camera_actual = R.from_matrix(R_target_camera_actual).as_euler('xyz')
        
        # Combine position and orientation for actual measurement
        camera_to_target_actual = np.concatenate([target_position_camera_actual, target_orientation_camera_actual])
        
        # === COMMANDED/EXPECTED MEASUREMENT (with nominal parameters) ===
        # Get commanded camera pose in world frame (with nominal parameters)
        joint_lengths_commanded = self.joint_lengths_nominal
        XOffsets_commanded = self.XNominal
        camera_pose_commanded = self.get_fk_calibration_model(
            joint_positions=joint_positions_commanded, 
            joint_lengths=joint_lengths_commanded, 
            XOffsets=XOffsets_commanded
        )
        camera_position_commanded = camera_pose_commanded[:3]
        camera_rotation_commanded = R.from_euler('xyz', camera_pose_commanded[3:6]).as_matrix()
        
        # Create homogeneous transformation from world to camera (commanded)
        T_world_to_camera_commanded = np.eye(4)
        T_world_to_camera_commanded[:3, :3] = camera_rotation_commanded.T  # Inverse rotation
        T_world_to_camera_commanded[:3, 3] = -camera_rotation_commanded.T @ camera_position_commanded
        
        # Transform target from world to camera frame (expected measurement)
        target_in_camera_commanded = T_world_to_camera_commanded @ target_homogeneous
        target_position_camera_commanded = target_in_camera_commanded[:3]
        
        # Calculate target orientation relative to camera frame (commanded)
        # Transform target orientation from world to camera frame
        # R_target_camera = R_camera_world^T @ R_target_world
        R_target_camera_commanded = camera_rotation_commanded.T @ R_target_world
        
        # Convert rotation matrix back to Euler angles
        target_orientation_camera_commanded = R.from_matrix(R_target_camera_commanded).as_euler('xyz')
        
        # Combine position and orientation for commanded measurement
        camera_to_target_commanded = np.concatenate([target_position_camera_commanded, target_orientation_camera_commanded])
        
        # Add measurement noise if desired
        # noise = np.random.uniform(-self.noiseMagnitude, self.noiseMagnitude, 6)
        # camera_to_target_actual += noise
        
        # Store camera-specific measurements for later processing
        # These represent the full 6DOF transformation from camera to target
        if hasattr(self, 'target_transforms_measured') and self.current_sample < self.n:
            self.target_transforms_measured[self.current_sample, :] = camera_to_target_actual
            self.target_transforms_expected[self.current_sample, :] = camera_to_target_commanded
            self.target_displacements_measured[self.current_sample, :] = target_position_camera_actual
            self.target_displacements_expected[self.current_sample, :] = target_position_camera_commanded
        else:
            # Fallback for single measurement or initialization
            self.target_transforms_measured = camera_to_target_actual.reshape(1, 6)
            self.target_transforms_expected = camera_to_target_commanded.reshape(1, 6)
            self.target_displacements_measured = target_position_camera_actual.reshape(1, 3)
            self.target_displacements_expected = target_position_camera_commanded.reshape(1, 3)
        
        worldToTargetActual = np.concatenate([target_positions_world,target_orientation_world])-self.dX
        self.targetArrayActual[self.current_sample][:] = worldToTargetActual
        self.targetArrayCommanded[self.current_sample][:] = pose_commanded

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

        self.targetArrayActual[self.current_sample][:] = pose_actual
        self.targetArrayCommanded[self.current_sample][:] = pose_commanded
      
        return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded
    
    def compute_jacobians(self, joint_angles, camera_to_target_measured=None):
        """Compute Jacobians for all measurements in current iteration
        
        Args:
            joint_angles: Joint angles for the current measurement
            camera_to_target_measured: 6DOF camera-to-target pose [px, py, pz, rx, ry, rz]
        """
        l = sp.symbols('l1:7')
        x = sp.symbols('x1:7')
        q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        
        # Variable list includes only robot parameters (no target error parameters)
        vars = list(self.q) + list(self.l) + list(self.x)
        
        num_measurements = 1
        numJacobianTrans = np.ones((3*num_measurements, len(vars)))
        rotCount = 9
        numJacobianRot = np.ones((rotCount*num_measurements, len(vars)))

        joint_lengths = self.joint_lengths_nominal + np.sum(self.dLMat, axis=0)
        XOffsets = self.XNominal + np.sum(self.dXMat, axis=0)
        
        # Prepare substitution dictionary (robot parameters only)
        subs_dict = {
            **{q[j]: joint_angles[j] for j in range(6)},  # Substitute q variables
            **{l[j]: joint_lengths[j] for j in range(6)},
            **{x[j]: XOffsets[j] for j in range(6)},
        }
        
        # For camera mode, also substitute the measured camera-to-target pose
        if self.camera_mode and camera_to_target_measured is not None:
            # Substitute measured target position and orientation
            subs_dict.update({
                self.measured_target_position[0]: camera_to_target_measured[0],  # tx
                self.measured_target_position[1]: camera_to_target_measured[1],  # ty
                self.measured_target_position[2]: camera_to_target_measured[2],  # tz
                self.measured_target_orientation[0]: camera_to_target_measured[3],  # rx_t
                self.measured_target_orientation[1]: camera_to_target_measured[4],  # ry_t
                self.measured_target_orientation[2]: camera_to_target_measured[5],  # rz_t
            })
        elif self.camera_mode:
            # Use zero camera-to-target transform as default
            subs_dict.update({
                self.measured_target_position[0]: 0.0,
                self.measured_target_position[1]: 0.0,
                self.measured_target_position[2]: 0.0,
                self.measured_target_orientation[0]: 0.0,
                self.measured_target_orientation[1]: 0.0,
                self.measured_target_orientation[2]: 0.0,
            })
        
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
            # For camera mode, use the symbolic forward kinematics to compute expected target positions
            # based on robot parameters and measured camera-to-target transformations
            
            translationDifferences = []
            rotationalDifferences = []
            
            for i in range(len(self.target_transforms_measured)):
                # Get the measured camera-to-target transformation
                camera_to_target_measured = self.target_transforms_measured[i]
                joint_positions = self.joint_positions_commanded[i]
                
                # Compute where the target SHOULD be using nominal robot parameters
                # and the measured camera-to-target transformation
                joint_lengths_nominal = self.joint_lengths_nominal
                XOffsets_nominal = self.XNominal
                expected_target_pose = self.get_fk_calibration_model(
                    joint_positions=joint_positions,
                    joint_lengths=joint_lengths_nominal,
                    XOffsets=XOffsets_nominal,
                    camera_to_target=camera_to_target_measured
                )
                
                # Compute where the target ACTUALLY is using actual robot parameters
                # and the measured camera-to-target transformation
                joint_lengths_actual = self.joint_lengths_actual
                XOffsets_actual = self.XActual.flatten()
                actual_target_pose = self.get_fk_calibration_model(
                    joint_positions=self.joint_positions_actual[i],
                    joint_lengths=joint_lengths_actual,
                    XOffsets=XOffsets_actual,
                    camera_to_target=camera_to_target_measured
                )
                
                # Ensure proper type conversion from SymPy to NumPy
                expected_target_pose = np.array(expected_target_pose, dtype=np.float64)
                actual_target_pose = np.array(actual_target_pose, dtype=np.float64)
                
                # Compute differences
                translation_diff = actual_target_pose[:3] - expected_target_pose[:3]
                translationDifferences.append(translation_diff)
                
                # Rotation differences using rotation matrices
                R_actual = R.from_euler('xyz', actual_target_pose[3:6]).as_matrix()
                R_expected = R.from_euler('xyz', expected_target_pose[3:6]).as_matrix()
                R_diff = R_actual - R_expected
                rotationalDifferences.append(R_diff.flatten())
            
            # Convert to NumPy arrays with proper dtype
            translationDifferences = np.array(translationDifferences, dtype=np.float64)
            rotationalDifferences = np.array(rotationalDifferences, dtype=np.float64).reshape(-1, 9)
        else:
            # Standard mode: pose differences
            measurements_actual = np.array(measurements_actual, dtype=np.float64)
            measurements_commanded = np.array(measurements_commanded, dtype=np.float64)
            
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
            rotationalDifferences = np.array(rotationalDifferences, dtype=np.float64).reshape(-1, 9)
    
        # Ensure proper dtype before adding
        translationDifferences = translationDifferences.astype(np.float64) 
        rotationalDifferences = rotationalDifferences.astype(np.float64) 
    
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
        
        # Use rotational information when available (camera mode with full transforms)
        if self.camera_mode:
            rotation_weight = 0.1     # Include rotational errors for camera mode
        else:
            rotation_weight = 0.0     # Standard mode - rotation often less reliable
        
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
            
        # Extract parameter updates (18 parameters: 6 joint + 6 length + 6 base)
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
            print("Camera mode: Using full 6DOF camera-to-target transformations")
            print("Translation errors (avg):", np.mean(np.abs(translationDifferences), axis=0))
            print("Rotation errors (avg):", np.mean(np.abs(rotationalDifferences), axis=0))
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
    # Test target orientation (20 degrees about z-axis, 10 degrees about x-axis)
    target_orientation_euler = np.array([0, 0, 0])  # [roll, pitch, yaw] in radians

    # Create simulator with target orientation
    simulator = CalibrationSimulator(n=8, numIters=10, camera_mode=True, dQMagnitude=0.1,
                                      dLMagnitude=0.0, dXMagnitude=0.0, 
                                     target_orientation=target_orientation_euler)
    simulator.target_positions_world = np.array([0.0, -0.3, 0])

    
    print(f"Target orientation (Euler): roll={np.degrees(target_orientation_euler[0]):.1f}°, "
          f"pitch={np.degrees(target_orientation_euler[1]):.1f}°, "
          f"yaw={np.degrees(target_orientation_euler[2]):.1f}°")
    
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
            
            # Get the measured camera-to-target transformation for Jacobian computation
            if simulator.camera_mode and hasattr(simulator, 'target_transforms_measured'):
                camera_to_target_measured = simulator.target_transforms_measured[simulator.current_sample-1]
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample-1],
                    camera_to_target_measured=camera_to_target_measured
                )
            else:
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample-1]
                )
            
            simulator.current_sample += 1  
            print(f"Measurement {i}: Generated")

        
        # Process all measurements for this iteration
        results = simulator.process_iteration_results(simulator.poseArrayActual, simulator.poseArrayCommanded,simulator.numJacobianTrans,simulator.numJacobianRot)
    
    # Save results to CSV
    #simulator.save_to_csv()
    
    print('done')

if __name__ == "__main__":
    main()