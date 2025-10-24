#!/usr/bin/env python3
import pickle
import numpy as np
from calibrationConvergenceSimulation import CalibrationConvergenceSimulator
from ar4_robot_py import AR4Robot
import rclpy

def load_simulator_data(filename='simulator_state.pkl'):
    """Load the saved simulator data and recreate simulator object"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Create a new simulator instance
    robot = AR4Robot()  # You might need to handle this differently
    simulator = CalibrationConvergenceSimulator(
        n=data['n'], 
        numIters=data['numIters'],
        dQMagnitude=data['dQMagnitude'],
        dLMagnitude=data['dLMagnitude'],
        dXMagnitude=data['dXMagnitude'],
        camera_mode=data['camera_mode'],
        noiseMagnitude=data['noiseMagnitude'],
        robot=robot
    )
    
    # Restore the saved state
    simulator.targetPosEst = data['targetPosEst']
    simulator.targetOrientEst = data['targetOrientEst']
    simulator.targetPosActual = data['targetPosActual']
    simulator.targetOrientActual = data['targetOrientActual']
    simulator.targetPosNom = data['targetPosNom']
    simulator.targetOrientNom = data['targetOrientNom']
    
    if data['targetPoseMeasured'] is not None:
        simulator.targetPoseMeasured = data['targetPoseMeasured']
    if data['targetPoseExpected'] is not None:
        simulator.targetPoseExpected = data['targetPoseExpected']
    if data['numJacobianTrans'] is not None:
        simulator.numJacobianTrans = data['numJacobianTrans']
    if data['numJacobianRot'] is not None:
        simulator.numJacobianRot = data['numJacobianRot']
    if data['joint_positions_commanded'] is not None:
        simulator.joint_positions_commanded = data['joint_positions_commanded']
    if data['joint_positions_actual'] is not None:
        simulator.joint_positions_actual = data['joint_positions_actual']
    if data['camera_to_target_meas_test'] is not None:
        simulator.camera_to_target_meas_test = data['camera_to_target_meas_test']
    if data['dX'] is not None:
        simulator.dX = data['dX']
    
    simulator.current_sample = data.get('current_sample', 0)
    simulator.current_iteration = data.get('current_iteration', 0)
    
    return simulator, data

# Example usage
if __name__ == "__main__":
    rclpy.init()
    simulator, raw_data = load_simulator_data()
    print(f"Loaded simulator with {simulator.n} measurements")
    print(f"Target position estimate: {simulator.targetPosEst}")
    
    # You can also access raw data directly
    print(f"Raw data keys: {raw_data.keys()}")