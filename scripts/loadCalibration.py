#!/usr/bin/env python3
import pickle
import numpy as np
from calibrationConvergenceSimulation import CalibrationConvergenceSimulator
from ar4_robot_py import AR4Robot
import rclpy

def save_simulator(simulator, filename='simulator_state.pkl'):
    simulator.robot = None
    simulator.fk_translation_func = None
    simulator.fk_rotation_func = None
    simulator.jacobian_translation_func = None  # This one was missing!
    simulator.jacobian_rotation_func = None     # This one was missing!
    simulator.jacobian_translation = None
    simulator.jacobian_rotation = None
    simulator.translation_vector = None
    simulator.rotation_matrix = None
    with open(filename, 'wb') as f:
            pickle.dump(simulator, f)

    print(f'Simulator state saved to {filename}')


def load_simulator(filename='simulator_state.pkl'):
    """Load the saved simulator data and recreate simulator object"""
    with open(filename, 'rb') as f:
        simulator = pickle.load(f)
    
    robot = AR4Robot()
    simulator.robot = robot
    return simulator

# Example usage
if __name__ == "__main__":
    rclpy.init()
    simulator = load_simulator()
    position = np.array([0.4, 0.0, 0.3])
    orientation_euler = np.array([0.0, -np.pi/2, 0.0])
    for _ in range(5):
        simulator.moveToPose(position, orientation_euler,calibrate = False)
        simulator.moveToPose(position, orientation_euler,calibrate = True)
    print("Finished loading and using simulator.")
