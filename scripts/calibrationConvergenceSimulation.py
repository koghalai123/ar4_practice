#!/usr/bin/env python3

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

class CalibrationConvergenceSimulator:
    def __init__(self, n=10, numIters=10, dQMagnitude=0.1, dLMagnitude=0.0,
                 dXMagnitude=0.1, camera_mode=False, noiseMagnitude=0.00, robot = None):
        self.camera_mode = camera_mode
        
        self.measuredErrorsHistory = np.ones((numIters, 2))
        self.estimatedTargetPoseHistory = np.ones((numIters, 6))
        self.measuredTargetPoseHistory = np.ones((numIters*n, 6))
        self.calibrationParameterHistory = np.ones((numIters, 18))

        if robot is not None:
            self.robot = robot
            self.targetPosNom, self.targetOrientNom = self.robot.from_preferred_frame(
            np.array([0.3,0,0]),np.array([np.pi,-np.pi/2,0]))
        else:
            self.targetPosNom = np.array([0,-0.3,0])
            self.targetOrientNom = np.array([-np.pi/2,0,np.pi])
            
        self.OnPhysicalRobot = False
        self.PhysicalCameraMeasurement = np.zeros(6)

        self.n=n
        self.m = 6
        self.noiseMagnitude = noiseMagnitude

        self.resetMatrices()
        
        self.dQMagnitude = dQMagnitude
        self.dQ = np.random.uniform(-self.dQMagnitude, self.dQMagnitude, (1, 6))[0]
        
        
        self.LMat = np.ones((1, 6))
        self.dLMagnitude = dLMagnitude
        self.dL = np.random.uniform(-self.dLMagnitude, self.dLMagnitude, (1, 6))[0]
        
        self.XNominal = np.zeros((6))
        self.dXMagnitude = dXMagnitude
        self.dX = np.random.uniform(-self.dXMagnitude, self.dXMagnitude, (1,6))[0]
        #self.dX[3:5]=0
        self.XActual = self.XNominal + self.dX
        
        self.numIters = numIters
        self.dQMat = np.zeros((self.numIters, 6))
        self.dLMat = np.zeros((self.numIters, 6))
        self.dXMat = np.zeros((self.numIters, 6))
        self.avgAccMat = np.ones((self.numIters,2))
        
        self.symbolic_matrices = self.loadSymbolicTransforms()
        self.baseToWrist = sp.eye(4)
        self.wristToBase = sp.eye(4)
                
        self.current_iter = 0
        self.current_sample = 0
        
        
        
        self.targetPosEst = self.targetPosNom
        self.targetOrientEst = self.targetOrientNom
        
        self.targetPosActual = self.targetPosNom + self.dX[:3]
        self.targetOrientActual = self.targetOrientNom
        
        
        self.setup_kinematics()
    
    def resetMatrices(self):
        self.poseArrayActual = np.zeros((self.n, self.m))
        self.poseArrayCommanded = np.zeros((self.n, self.m))
        self.joint_positions_actual = np.zeros((self.n, self.m))
        self.joint_positions_commanded = np.zeros((self.n, self.m))
        self.poseArrayCalibrated = np.zeros((self.n, self.m))
        
        self.targetPoseExpected = np.zeros((self.n, self.m))
        self.targetPoseMeasured = np.zeros((self.n, self.m))
        
        numParams=18
        self.numJacobianTrans = np.zeros((0, numParams))
        self.numJacobianRot = np.zeros((0, numParams))

    def loadSymbolicTransforms(self):
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
    
        
    def get_homogeneous_transform(self, xyz, euler_angles, rotation_order='xyz'):
        rotation = R.from_euler(rotation_order, euler_angles)
        rot_mat = rotation.as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = xyz
        
        return transform
        
    def symbolic_transform_with_ref_frames(self, xyz, euler_angles, rotation_order='XYZ'):

        rx_t, ry_t, rz_t = euler_angles
        cx, cy, cz = sp.cos(rx_t), sp.cos(ry_t), sp.cos(rz_t)
        sx, sy, sz = sp.sin(rx_t), sp.sin(ry_t), sp.sin(rz_t)
        R_x = sp.Matrix([[1, 0, 0],
                            [0, cx, -sx],
                            [0, sx, cx]])
        R_y = sp.Matrix([[cy, 0, sy],
                            [0, 1, 0],
                            [-sy, 0, cy]])
        R_z = sp.Matrix([[cz, -sz, 0],
                            [sz, cz, 0],
                            [0, 0, 1]])
        R_ct = (R_z * R_y * R_x)
        T = sp.eye(4)
        T[:3, :3] = R_ct
        T[:3, 3] = xyz
        
        return T
        
    def setup_kinematics(self):
        """Setup symbolic kinematic chain and measurement model
        
        Unified measurement approach for both standard and camera modes:
        
        Standard Mode:
        - Jacobian: ∂(robot_end_effector_pose)/∂(parameters)
        - Expected: FK(commanded_joints + estimated_corrections, nominal_lengths + estimated_corrections, nominal_offsets + estimated_corrections)
        - Actual: FK(commanded_joints + true_errors, true_lengths, true_offsets)
        - Error: actual - expected
        
        Camera Mode:
        - Jacobian: ∂(target_pose_in_world)/∂(parameters) 
        - Expected: FK_world_to_target(commanded_joints + estimated_corrections, nominal_lengths + estimated_corrections, nominal_offsets + estimated_corrections, measured_camera_to_target)
        - Actual: true_target_pose_in_world (ground truth)
        - Error: actual - expected
        
        Both modes use the same least squares formulation: find parameter corrections that minimize (actual - expected)
        No sign corrections needed because measurement directions are consistent.
        """
        
        
        self.l = sp.symbols('l1:7')
        self.x = sp.symbols('x1:7')
        self.q = sp.symbols('q_joint_1 q_joint_2 q_joint_3 q_joint_4 q_joint_5 q_joint_6')
        self.camera_measurements = sp.symbols('c1:7')
        
        #self.originToBase = self.symbolic_transform_with_ref_frames(self.x[0:3], [self.x[3], self.x[4], 0], rotation_order='XYZ')
        self.originToBase = self.symbolic_transform_with_ref_frames([self.x[0], self.x[1], 0.0], self.x[3:], rotation_order='XYZ')
        self.originToBaseActual = self.get_homogeneous_transform([self.XActual[0],self.XActual[1],0.0], [0,0,0], rotation_order='xyz')

        for i in range(1, 7):
            key = "Joint" + str(i)
            symbolic_matrix = self.symbolic_matrices[key]
            symbolic_matrix = self.symbolic_matrices[key]
            translation_vector = symbolic_matrix[:3, 3]
            norm_symbolic = sp.sqrt(sum(component**2 for component in translation_vector))
            self.LMat[0, i - 1] = norm_symbolic
            symbolic_matrix[:3,3]=symbolic_matrix[:3,3]#*self.l[i-1]
            self.baseToWrist = self.baseToWrist * symbolic_matrix
            self.wristToBase = symbolic_matrix.inv() * self.wristToBase
            self.symbolic_matrices[key] = symbolic_matrix
        
        self.originToWrist = self.originToBase*self.baseToWrist
        
        if self.camera_mode:
            self.measured_target_position = self.camera_measurements[:3]
            self.measured_target_orientation = self.camera_measurements[3:]

            self.cameraToTarget = self.symbolic_transform_with_ref_frames(
                self.measured_target_position, self.measured_target_orientation
            )
            self.originToTarget = self.originToWrist * self.cameraToTarget
            self.translation_vector = self.originToTarget[:3, 3]
            self.rotation_matrix = self.originToTarget[:3, :3]
        else:
            self.translation_vector = self.originToWrist[:3, 3]
            self.rotation_matrix = self.originToWrist[:3, :3]
            self.originToTarget = self.originToWrist

        # CREATE THE JACOBIANS FIRST (this was missing!)
        vars = list(self.q) + list(self.l) + list(self.x)
        
        self.jacobian_translation = self.translation_vector.jacobian(vars)
        
        self.rotation_matrix_flat = self.rotation_matrix.reshape(9, 1)
        self.jacobian_rotation = self.rotation_matrix_flat.jacobian(vars)
        
        self.joint_lengths_nominal = np.ones(6)
        self.joint_lengths_actual = self.joint_lengths_nominal + self.dL.flatten()

        # NOW create the lambdify functions
        vars_list = list(self.q) + list(self.l) + list(self.x)
        if self.camera_mode:
            vars_list += list(self.camera_measurements)
                    
        # Convert the JACOBIANS to fast numpy functions
        self.jacobian_translation_func = sp.lambdify(
            vars_list, self.jacobian_translation, 'numpy', cse=True
        )
        
        self.jacobian_rotation_func = sp.lambdify(
            vars_list, self.jacobian_rotation, 'numpy', cse=True
        )
        
        # Convert FK to numerical functions (for forward kinematics only)
        self.fk_translation_func = sp.lambdify(
            vars_list, self.translation_vector, 'numpy', cse=True
        )
        self.fk_rotation_func = sp.lambdify(
            vars_list, self.rotation_matrix, 'numpy', cse=True
        )
    
    def set_current_iteration(self, iteration_index):
        """Set the current iteration index"""
        if iteration_index >= 0 and iteration_index < self.numIters:
            self.current_iter = iteration_index
        else:
            print(f"Warning: Invalid iteration index {iteration_index}. Using 0 instead.")
            self.current_iter = 0
    
    def get_fk_calibration_model(self, joint_positions, joint_lengths, XOffsets, camera_to_target=None):
        """Fast numerical FK computation using lambdify"""
        
        if self.camera_mode:
            if camera_to_target is None:
                camera_to_target = np.zeros(6)
            args = list(joint_positions) + list(joint_lengths) + list(XOffsets) + list(camera_to_target)
        else:
            args = list(joint_positions) + list(joint_lengths) + list(XOffsets)
        
        # Evaluate numerically
        translation = self.fk_translation_func(*args)
        rotation_matrix = self.fk_rotation_func(*args)
        
        # Convert rotation matrix to Euler angles
        euler_angles = R.from_matrix(np.array(rotation_matrix).astype(np.float64)).as_euler('xyz')
        
        return np.concatenate([translation.flatten(), euler_angles])
    
    def generate_measurement_pose(self, robot, pose = None, calibrate=True, frame = "end_effector_link", camera_to_target_meas=None):

        acceptRandom = False
        if pose is None:
            pose = np.random.uniform(-0.3, 0.3, (1, 6))[0]
            acceptRandom= True
        position = pose[:3]
        orientation =  pose[3:6]
        #transformed_position, transformed_orientation = robot.fromMyPreferredFrame(position, orientation, old_reference_frame=frame, new_reference_frame="base_link")
        
        #pos,ori = robot.get_current_pose()
        joint_positions_ik = robot.get_ik(position=position, euler_angles=orientation, frame_id=frame)
        if joint_positions_ik is not None or acceptRandom:
            #robot.get_fk(joint_positions_ik)
            if self.camera_mode:
                pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints_camera(joint_positions_ik, calibrate,camera_to_target_meas)
            else:
                pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints(joint_positions_ik, calibrate)
        
        
            return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded
        else:
            return None, None, None, None
        
    def generate_measurement_joints_camera(self, joint_positions_input=None, calibrate=True, camera_to_target_meas=None):
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
        #
        pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = self.generate_measurement_joints(joint_positions_commanded=joint_positions_input, calibrate=calibrate)
        
        targetPosActual = self.targetPosActual
        
        joint_lengths = self.joint_lengths_actual
        #XOffsets = self.XActual.flatten()
        camera_pose_actual = self.get_fk_calibration_model(
        joint_positions=joint_positions_actual, 
        joint_lengths=joint_lengths, 
        XOffsets=self.XNominal,
        camera_to_target = np.zeros(6)
        )
        camera_position_actual = camera_pose_actual[:3]
        camera_rotation_actual = R.from_euler('xyz', camera_pose_actual[3:6]).as_matrix()

        R_target_world = R.from_euler('xyz', self.targetOrientActual).as_matrix()
        R.from_matrix(R_target_world[:3,:3]).as_euler('xyz')
        R.from_euler('xyz', R.from_matrix(R_target_world[:3,:3]).as_euler('xyz')).as_matrix()
        T_world_to_target = np.eye(4)
        T_world_to_target[:3, 3] = targetPosActual
        T_world_to_target[:3, :3] = R_target_world

        T_world_to_camera_actual = np.eye(4)
        T_world_to_camera_actual[:3, :3] = camera_rotation_actual
        T_world_to_camera_actual[:3, 3] = camera_position_actual
        T_camera_target_actual = np.linalg.inv(T_world_to_camera_actual) @ T_world_to_target

        R_target_camera_actual = T_camera_target_actual[:3, :3]
        target_orientation_camera_actual = R.from_matrix(R_target_camera_actual).as_euler('xyz')
        target_position_camera_actual = T_camera_target_actual[:3, 3]

        camera_to_target_meas_test = 0.2*self.noiseMagnitude*np.random.uniform(-1, 1, 6) + np.concatenate([target_position_camera_actual, target_orientation_camera_actual])

        self.camera_to_target_meas_test = camera_to_target_meas_test

        joint_lengths_commanded = self.joint_lengths_nominal
        XOffsets_commanded = self.XNominal
        camera_pose_commanded = self.get_fk_calibration_model(
            joint_positions=joint_positions_commanded, 
            joint_lengths=joint_lengths_commanded, 
            XOffsets=XOffsets_commanded,
            camera_to_target = np.zeros(6)
        )
        camera_position_commanded = camera_pose_commanded[:3]
        camera_rotation_commanded = R.from_euler('xyz', camera_pose_commanded[3:6]).as_matrix()

        # Create camera-to-world transformation 
        T_world_to_camera_commanded = np.eye(4)
        T_world_to_camera_commanded[:3, :3] = camera_rotation_commanded
        T_world_to_camera_commanded[:3, 3] = camera_position_commanded

        T_camera_target_commanded = np.linalg.inv(T_world_to_camera_commanded) @ T_world_to_target
        target_position_camera_commanded = T_camera_target_commanded[:3, 3]
        R_target_camera_commanded = T_camera_target_commanded[:3, :3]

        target_orientation_camera_commanded = R.from_matrix(R_target_camera_commanded).as_euler('xyz')
        camera_to_target_commanded = np.concatenate([target_position_camera_commanded, target_orientation_camera_commanded])

        joint_lengths_est = self.joint_lengths_nominal + np.sum(self.dLMat, axis=0)
        #XOffsets_est = self.XNominal + np.sum(self.dXMat, axis=0)
        
        T_world_to_camera_commanded @ T_camera_target_commanded

        T_world_to_target_from_calc = T_world_to_camera_actual @ T_camera_target_actual
        R.from_matrix(T_world_to_target_from_calc[:3,:3]).as_euler('xyz')
        
        if camera_to_target_meas is None:
            self.camera_to_target_meas = camera_to_target_meas_test
            camera_to_target_meas = camera_to_target_meas_test
        worldToTargetMeasured = self.get_fk_calibration_model(joint_positions = joint_positions_commanded,
                                                                 joint_lengths = joint_lengths_est,
                                                                 XOffsets = self.XNominal,
                                                                 camera_to_target = camera_to_target_meas)
        '''worldToCameraFocus = self.get_fk_calibration_model(joint_positions = joint_positions_commanded,
                                                                 joint_lengths = joint_lengths_est,
                                                                 XOffsets = XOffsets_est,
                                                                 camera_to_target = np.array([0,0,camera_to_target_meas[2],0,0,0]))'''
                                                                 
                                                                 
        #self.cameraFocus = worldToCameraFocus
        worldToTargetEst = np.concatenate([self.targetPosEst, self.targetOrientEst])

        # Store measurements: expected vs actual (consistent with standard mode direction)
        self.targetPoseExpected[self.current_sample][:] = worldToTargetEst
        self.targetPoseMeasured[self.current_sample][:] = worldToTargetMeasured

        return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded

    def generate_measurement_joints(self, joint_positions_commanded = None, calibrate=True):
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
            if self.camera_mode:
                XOffsets_est = self.XNominal
                self.targetPosEst = self.targetPosNom + np.sum(self.dXMat, axis=0)[:3]
                self.targetOrientEst = self.targetOrientNom + np.sum(self.dXMat, axis=0)[3:]
            else:
                XOffsets_est = self.XNominal #+ np.sum(self.dXMat, axis=0)
            # Apply current parameter estimates to get expected measurement
            joint_lengths_est = self.joint_lengths_nominal + np.sum(self.dLMat, axis=0)
            
            joint_positions_est = joint_positions_commanded - np.sum(self.dQMat, axis=0)
            pose_commanded = self.get_fk_calibration_model(joint_positions_commanded, joint_lengths_est, XOffsets_est)
        
        joint_lengths = self.joint_lengths_actual
        XOffsets =  self.XActual.flatten()
        joint_positions_actual = joint_positions_est + self.dQ + np.random.uniform(-self.noiseMagnitude, self.noiseMagnitude, (1, 6))[0]
        pose_actual = self.get_fk_calibration_model(joint_positions_actual, joint_lengths, XOffsets)


        self.poseArrayActual[self.current_sample][:] = pose_actual
        self.joint_positions_actual[self.current_sample][:] = joint_positions_actual        
        self.poseArrayCommanded[self.current_sample][:] = pose_commanded
        
        self.targetPoseExpected[self.current_sample][:] = pose_commanded
        self.targetPoseMeasured[self.current_sample][:] = pose_actual
        
                    
        return pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded

    def moveToPose(self, position, orientation, calibrate=True):
        frame = "base_link"
        joint_positions_ik = self.robot.get_ik(position=position, euler_angles=orientation, frame_id=frame)
        if joint_positions_ik is None:
            print("IK solution not found for the given pose.")
            return

        if calibrate:
            joint_positions = joint_positions_ik - np.sum(self.dQMat, axis=0) + self.dQ
        else:
            joint_positions = joint_positions_ik + self.dQ

        self.robot.move_to_joint_positions(joint_positions)


    def compute_jacobians(self, joint_angles, camera_to_target=None):
        """Fast numerical Jacobian computation using lambdify"""
        
        joint_lengths = self.joint_lengths_nominal
        XOffsets = self.XNominal
        
        if self.camera_mode:
            if camera_to_target is None:
                camera_to_target = np.zeros(6)
            args = list(joint_angles) + list(joint_lengths) + list(XOffsets) + list(camera_to_target)
        else:
            args = list(joint_angles) + list(joint_lengths) + list(XOffsets)
        
        # Evaluate the pre-computed Jacobians directly
        numJacobianTrans_raw = self.jacobian_translation_func(*args)
        numJacobianRot_raw = self.jacobian_rotation_func(*args)
        
        # Convert to numpy arrays with correct shape
        numJacobianTrans = np.array(numJacobianTrans_raw, dtype=np.float64)
        numJacobianRot = np.array(numJacobianRot_raw, dtype=np.float64)
        
        # Debug the shapes
        #print(f"Raw Jacobian shapes: Trans={numJacobianTrans.shape}, Rot={numJacobianRot.shape}")
        
        # The symbolic jacobians should give us the right shape directly
        # But if they're transposed, fix them:
        if numJacobianTrans.shape == (18, 3):
            numJacobianTrans = numJacobianTrans.T
        if numJacobianRot.shape == (18, 9):
            numJacobianRot = numJacobianRot.T
            
        ## Ensure final shape is correct
        assert numJacobianTrans.shape == (3, 18), f"Translation Jacobian wrong shape: {numJacobianTrans.shape}"
        assert numJacobianRot.shape == (9, 18), f"Rotation Jacobian wrong shape: {numJacobianRot.shape}"
        
        #print(f"Final Jacobian shapes: Trans={numJacobianTrans.shape}, Rot={numJacobianRot.shape}")
        
        # Store for iteration processing (same as slow version)
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
        """Compute differences between actual and commanded poses
        
        For both standard and camera modes:
        - measurements_commanded: Expected pose using current parameter estimates
        - measurements_actual: Actual measured pose (ground truth)
        - Difference = actual - expected (what we observe minus what we predict)
        - Jacobian shows how expected pose changes with parameters
        - Least squares finds parameter changes to minimize (actual - expected)
        """
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
        rotation_weight = 1.0     # Weight for rotational errors
        
        # Scale translational and rotational differences
        scaled_translation_differences = translation_weight * translationDifferences.flatten()
        scaled_rotational_differences = rotation_weight * rotationalDifferences.ravel()

        bMat = np.concatenate((scaled_translation_differences, scaled_rotational_differences))
        AMat = np.vstack((translation_weight * numJacobianTrans, rotation_weight * numJacobianRot))
        
        #bMat = scaled_translation_differences
        #AMat = translation_weight * numJacobianTrans

        
        errorEstimates, residuals, rank, singular_values = np.linalg.lstsq(AMat, bMat, rcond=None)
        
        # Apply maximum caps to the parameter estimates
        max_dQ_cap = 0.2
        max_dL_cap = 5
        max_dX_cap = 0.4
        
        # Cap the estimates
        j = self.current_iter
        dLIndex = 12
        if j==0:
            #Skip the cap on J6 on the first iteration, since there can be a lot of variability from the end effector
            dLIndex = 11
        errorEstimates[0:6] = np.clip(errorEstimates[0:6], -max_dQ_cap, max_dQ_cap)    # Joint corrections
        errorEstimates[6:dLIndex] = np.clip(errorEstimates[6:dLIndex], -max_dL_cap, max_dL_cap)  # Length corrections
        errorEstimates[12:18] = np.clip(errorEstimates[12:18], -max_dX_cap, max_dX_cap) # Base offset corrections
        
        return errorEstimates
    
    def process_iteration_results(self, measurements_actual, measurements_expected,numJacobianTrans,numJacobianRot):
        """Process all measurements for the current iteration"""
        j = self.current_iter

        translationDifferences, rotationalDifferences = self.compute_differences(measurements_expected, measurements_actual)
        
        avgAccuracyError, avgRotationalError, avgTransAndRotError = self.compute_error_metrics(
            translationDifferences, rotationalDifferences)
        self.avgAccMat[j,:] = avgTransAndRotError
        
        errorEstimates = self.compute_calibration_parameters(
            translationDifferences, rotationalDifferences, numJacobianTrans, numJacobianRot)
        
        dQEst = errorEstimates[0:6]
        self.dQMat[j, :] = dQEst

        dLEst = errorEstimates[6:12]
        self.dLMat[j, :] = dLEst
        
        dXEst = errorEstimates[12:18]
        self.dXMat[j, :] = np.concatenate([-dXEst[:3],-dXEst[3:]])
        #self.dXMat[j, 0], self.dXMat[j, 1], self.dXMat[j, 2] = self.dXMat[j, 1], -self.dXMat[j, 0], -self.dXMat[j, 2]

        print("Iteration: ", j)
        print("Avg Pose Error: ", avgTransAndRotError)
        print("dLAct: ", self.dL)
        print("dLEst: ", np.sum(self.dLMat,axis=0))
        print("dQAct: ", self.dQ)
        print("dQEst: ", np.sum(self.dQMat,axis=0))
        print("dXAct: ", self.dX)
        print("dXEst: ", np.sum(self.dXMat,axis=0))
        print("")
        
        
        self.measuredErrorsHistory[j,:] = avgTransAndRotError
        self.estimatedTargetPoseHistory[j,:] = self.targetPoseExpected[0,:]
        self.measuredTargetPoseHistory[self.n*(j):self.n*(j+1),:] = measurements_actual
        temp = np.hstack((self.dQMat,self.dLMat,self.dXMat))
        self.calibrationParameterHistory = np.cumsum(temp, axis=0)
        
        self.resetMatrices()
        self.current_sample = 0
        
        return avgTransAndRotError, np.sum(self.dLMat,axis=0), self.dQ, np.sum(self.dQMat,axis=0), np.sum(self.dXMat,axis=0)
    
    def get_new_end_effector_position(self):
        robot=self.robot
        random_numbers = np.random.rand(3) 
        scale = 0.25
        random_pos = (random_numbers-0.5)*scale
        xOffset = 0 - scale
        yOffset = 0
        zOffset = robot.pos_offsets["z"] -scale*1.2
        x = xOffset + random_pos[0]
        y = yOffset + random_pos[1]
        z = zOffset + random_pos[2]
        relativeToHomeAtGround = np.array([x,y,z])

        globalHomePos = np.array([robot.pos_offsets["x"],robot.pos_offsets["y"],robot.pos_offsets["z"]])
        return relativeToHomeAtGround, random_pos, globalHomePos
    def calculate_camera_pose(self,x, y, z, roll_offset=0.0, pitch_offset=0.0, yaw_offset=0.0):
        # Calculate robot end effector orientation to point toward target
        # Vector [x, y, z] is from end effector to target in world frame
        
        # Vector from end effector to target
        vector_to_target = np.array([x, y, z])
        distance = np.linalg.norm(vector_to_target)
        
        if distance == 0:
            return 0.0, 0.0, 0.0
        
        # Calculate yaw: rotation around Z axis to point in XY direction of target
        yaw = np.arctan2(y, x)
        
        # Calculate pitch: elevation angle from XY plane
        xy_distance = np.sqrt(x**2 + y**2)
        pitch = np.arctan2(z, xy_distance)
        
        # Keep roll at 0 (no rotation around the pointing direction)
        roll = 0


        roll = np.arctan2(y, -z)#+np.pi/2
        temp = np.sqrt(z**2 + y**2)
        pitch = np.arctan2(x, temp) -np.pi/2
        yaw = 0#np.pi/2
        
        # Apply offsets
        roll += roll_offset
        pitch += pitch_offset
        yaw += yaw_offset
        
        return roll, pitch, yaw
        
    def save_to_csv(self, filename='calibrationData.csv'):
        """Save calibration data to CSV file"""
        j = self.current_iter
        combined = np.hstack((
            self.measuredErrorsHistory[:j+1, :],
            self.estimatedTargetPoseHistory[:j+1, :],
            self.calibrationParameterHistory[:j+1, :]
        ))

        # Create column names
        '''columns = (
            [f"avgError_{i}" for i in range(self.measuredErrorsHistory.shape[1])] +
            [f"estTargetPose_{i}" for i in range(self.estimatedTargetPoseHistory.shape[1])] +
            [f"calibParam_{i}" for i in range(self.calibrationParameterHistory.shape[1])]
        )'''
        
        columns = (
            ["Position Error", "Orientation Error"] +
            ["Estimated Target X", "Estimated Target Y", "Estimated Target Z", "Estimated Target Roll", "Estimated Target Pitch", "Estimated Target Yaw"] +
            ["dQ Estimated_1", "dQ Estimated_2", "dQ Estimated_3", "dQ Estimated_4", "dQ Estimated_5", "dQ Estimated_6"] +
            ["dL Estimated_1", "dL Estimated_2", "dL Estimated_3", "dL Estimated_4", "dL Estimated_5", "dL Estimated_6"] +
            ["dX Estimated_1", "dX Estimated_2", "dX Estimated_3", "dX Estimated_4", "dX Estimated_5", "dX Estimated_6"]
        )

        df = pd.DataFrame(combined, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

        # Optionally, save per-sample measured target poses as a separate file
        pd.DataFrame(self.measuredTargetPoseHistory).to_csv('measuredTargetPoseHistory.csv', index=False)


def main(args=None):
    # Create simulator
    rclpy.init()
    robot = AR4Robot()
    robot.disable_logging()
    simulator = CalibrationConvergenceSimulator(n=20, numIters=12, 
                dQMagnitude=0.1, dLMagnitude=0.0,
                 dXMagnitude=0.1, camera_mode=True, noiseMagnitude=0.0, robot = robot)
    frame = "end_effector_link"

    
    # Process each iteration separately
    
    if simulator.camera_mode:
        simulator.targetPosNom, simulator.targetOrientNom = simulator.robot.from_preferred_frame(
            np.array([0.3,0,0]),np.array([np.pi,-np.pi/2,0]))
        simulator.targetPosActual = simulator.targetPosNom + simulator.dX[:3]
        simulator.targetOrientActual = simulator.targetOrientNom + simulator.dX[3:]
        simulator.targetPosEst = simulator.targetPosNom
        simulator.targetOrientEst = simulator.targetOrientNom
    else:
        simulator.targetPosEst = np.array([0.0,0,0])
        simulator.targetOrientEst = np.array([0,0,0])
        simulator.targetPosActual = simulator.targetPosNom + simulator.dX[:3]
        simulator.targetOrientActual = simulator.targetOrientNom
        simulator.targetPosEst = simulator.targetPosNom
        simulator.targetOrientEst = simulator.targetOrientNom
    
    for j in range(simulator.numIters):
        print(f"\n--- Starting Iteration {j} ---")
        simulator.set_current_iteration(j)
                
        # Generate individual measurements
        for i in range(simulator.n):
            #pose_desired = np.array([ 2.97917633e-01,  1.18078317e-03,  4.61091580e-01, -2.56083746e-03,-1.56628020e+00,  0.00000000e+00])
            
            #pose_desired = np.array([ 0.37192116, -0.01913936,  0.48370802,  0.03954736, -1.7182883 , 0.        ])
            pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = simulator.generate_measurement_pose(
                robot=robot, calibrate=True, frame="base_link"
            )
            
            if simulator.camera_mode:
                # Use the actual camera-to-target measurement
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample], 
                    camera_to_target=simulator.camera_to_target_meas)
            else:
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample])
                
            simulator.current_sample += 1 
            print(f"Measurement {i}: Generated")

        results = simulator.process_iteration_results(
                simulator.targetPoseMeasured,
                simulator.targetPoseExpected,
                simulator.numJacobianTrans,
                simulator.numJacobianRot)
        simulator.save_to_csv()
    # Save results to CSV
    
    
    print('done')

if __name__ == "__main__":
    main()