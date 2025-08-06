#!/usr/bin/env python3
"""
Test script for camera-based calibration simulation

This script demonstrates how to use the calibration simulator in camera mode,
where measurements come from a camera mounted on the end effector observing
known targets and measuring displacement vectors.
"""

import rclpy
from ar4_robot_py import AR4Robot
from calibrationConvergenceSimulation import CalibrationSimulator
import numpy as np

def test_camera_calibration():
    """Test the calibration simulation using camera-based measurements"""
    
    
    try:
        # Create simulator with camera mode enabled
        print("Creating camera-based calibration simulator...")
        simulator = CalibrationSimulator(
            n=8,                    # Number of measurements per iteration
            numIters=10,            # Number of calibration iterations
            dQMagnitude=0.1,        # Joint angle error magnitude
            dLMagnitude=0.02,       # Link length error magnitude (small for realism)
            dXMagnitude=0.05,       # Base frame error magnitude
            camera_mode=True,       # Enable camera mode
            dCMagnitude=0.01        # Camera parameter error magnitude
        )
        
        # Create robot interface
        robot = AR4Robot()
        robot.disable_logging()
        
        print(f"Camera mode enabled: {simulator.camera_mode}")
        print(f"Total parameters to estimate: {simulator.numParameters}")
        print(f"Parameter breakdown:")
        print(f"  - Joint errors (dQ): 6 parameters")
        print(f"  - Link length errors (dL): 6 parameters") 
        print(f"  - Base frame errors (dX): 6 parameters")
        print(f"  - Camera transformation errors (dC): 6 parameters")
        print()
        
        # Print actual error values that we're trying to estimate
        print("Actual systematic errors to be estimated:")
        print(f"  Joint errors (dQ): {simulator.dQ}")
        print(f"  Link errors (dL): {simulator.dL}")
        print(f"  Base errors (dX): {simulator.dX}")
        print(f"  Camera errors (dC): {simulator.dC}")
        print()
        
        # Generate some sample target positions in camera frame
        # These represent known calibration targets that the camera can see
        target_positions = np.array([
            [0.05, 0.05, 0.1],    # Target 1: 5cm right, 5cm up, 10cm forward
            [-0.03, 0.02, 0.08],  # Target 2: 3cm left, 2cm up, 8cm forward
            [0.02, -0.04, 0.12],  # Target 3: 2cm right, 4cm down, 12cm forward
            [-0.01, -0.03, 0.09], # Target 4: 1cm left, 3cm down, 9cm forward
            [0.04, 0.01, 0.11],   # Target 5: 4cm right, 1cm up, 11cm forward
            [-0.02, 0.03, 0.07],  # Target 6: 2cm left, 3cm up, 7cm forward
            [0.01, -0.02, 0.13],  # Target 7: 1cm right, 2cm down, 13cm forward
            [0.03, -0.01, 0.06],  # Target 8: 3cm right, 1cm down, 6cm forward
        ])
        
        # Process each calibration iteration
        for j in range(simulator.numIters):
            print(f"\n=== Starting Camera Calibration Iteration {j+1}/{simulator.numIters} ===")
            simulator.set_current_iteration(j)
            
            # Generate random joint positions for measurements
            # In practice, these would be chosen to provide good coverage of workspace
            joint_positions_commanded = np.random.uniform(-2.0, 2.0, (simulator.n, 6))
            
            # Generate camera-based measurements
            print("Generating camera measurements...")
            joint_positions, camera_measurements = simulator.generate_measurement_joints_camera(
                joint_positions_commanded=joint_positions_commanded,
                target_positions=target_positions
            )
            
            print(f"Generated {len(camera_measurements)} camera displacement measurements")
            
            # For each measurement, compute the Jacobian
            print("Computing Jacobians for camera measurements...")
            for i in range(simulator.n):
                # Use the actual joint positions (with errors) for Jacobian computation
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_actual[i]
                )
                print(f"  Jacobian computed for measurement {i+1}")
            
            # Convert camera measurements to pose format for processing
            # In camera mode, we need to convert displacement measurements to poses
            measurements_actual = np.zeros((simulator.n, 6))
            measurements_commanded = np.zeros((simulator.n, 6))
            
            for i in range(simulator.n):
                # For camera mode, the "actual" measurement is the displacement vector
                # The "commanded" would be zero displacement (perfect positioning)
                measurements_actual[i, :3] = camera_measurements[i]  # Displacement in camera frame
                measurements_actual[i, 3:] = [0, 0, 0]  # No rotation measurement for now
                
                measurements_commanded[i, :] = [0, 0, 0, 0, 0, 0]  # Perfect positioning (no displacement)
            
            # Process the iteration results
            print("Processing iteration results...")
            results = simulator.process_iteration_results(
                measurements_actual, 
                measurements_commanded,
                simulator.numJacobianTrans,
                simulator.numJacobianRot
            )
            
            avgTransAndRotError, dLEst, dQAct, dQEst, dXEst, dCEst = results
            
            print(f"Results for iteration {j+1}:")
            print(f"  Average position error: {avgTransAndRotError[0]:.6f} m")
            print(f"  Average rotation error: {avgTransAndRotError[1]:.6f} rad")
            
            # Show convergence progress
            joint_error_progress = np.linalg.norm(dQEst - simulator.dQ)
            base_error_progress = np.linalg.norm(dXEst - simulator.dX)
            camera_error_progress = np.linalg.norm(dCEst - simulator.dC) if dCEst is not None else 0
            
            print(f"  Joint parameter estimation error: {joint_error_progress:.6f}")
            print(f"  Base parameter estimation error: {base_error_progress:.6f}")
            print(f"  Camera parameter estimation error: {camera_error_progress:.6f}")
        
        # Final results summary
        print(f"\n=== Final Camera Calibration Results ===")
        print("True vs Estimated Parameters:")
        print(f"Joint errors (dQ):")
        print(f"  True:      {simulator.dQ}")
        print(f"  Estimated: {np.sum(simulator.dQMat, axis=0)}")
        print(f"  Error:     {np.sum(simulator.dQMat, axis=0) - simulator.dQ}")
        
        print(f"Base errors (dX):")
        print(f"  True:      {simulator.dX}")
        print(f"  Estimated: {np.sum(simulator.dXMat, axis=0)}")
        print(f"  Error:     {np.sum(simulator.dXMat, axis=0) - simulator.dX}")
        
        print(f"Camera errors (dC):")
        print(f"  True:      {simulator.dC}")
        print(f"  Estimated: {np.sum(simulator.dCMat, axis=0)}")
        print(f"  Error:     {np.sum(simulator.dCMat, axis=0) - simulator.dC}")
        
        # Save results to CSV
        simulator.save_to_csv('camera_calibration_results.csv')
        print(f"\nResults saved to 'camera_calibration_results.csv'")
        
        # Calculate final accuracy metrics
        final_joint_accuracy = np.linalg.norm(np.sum(simulator.dQMat, axis=0) - simulator.dQ)
        final_base_accuracy = np.linalg.norm(np.sum(simulator.dXMat, axis=0) - simulator.dX)
        final_camera_accuracy = np.linalg.norm(np.sum(simulator.dCMat, axis=0) - simulator.dC)
        
        print(f"\nFinal Parameter Estimation Accuracy:")
        print(f"  Joint parameters: {final_joint_accuracy:.6f} rad RMS")
        print(f"  Base parameters: {final_base_accuracy:.6f} m/rad RMS")
        print(f"  Camera parameters: {final_camera_accuracy:.6f} m/rad RMS")
        
        if final_joint_accuracy < 0.01 and final_camera_accuracy < 0.005:
            print("✓ Camera calibration converged successfully!")
        else:
            print("⚠ Camera calibration needs more iterations or measurements")
        
    except Exception as e:
        print(f"Error during camera calibration test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'robot' in locals():
            robot.shutdown()
        rclpy.shutdown()

def compare_standard_vs_camera_mode():
    """Compare calibration performance between standard and camera modes"""
    
    print("\n=== Comparison: Standard vs Camera Mode ===")
    
    # Test parameters
    test_params = {
        'n': 6,
        'numIters': 8,
        'dQMagnitude': 0.08,
        'dLMagnitude': 0.01,
        'dXMagnitude': 0.03
    }
    
    results = {}
    
    for mode_name, camera_mode in [("Standard", False), ("Camera", True)]:
        print(f"\nTesting {mode_name} Mode...")
        
        # Create simulator
        if camera_mode:
            simulator = CalibrationSimulator(camera_mode=True, dCMagnitude=0.008, **test_params)
        else:
            simulator = CalibrationSimulator(camera_mode=False, **test_params)
        
        # Run a few iterations (simplified)
        final_errors = []
        for j in range(3):  # Just 3 iterations for comparison
            simulator.set_current_iteration(j)
            
            if camera_mode:
                # Generate camera measurements
                joint_pos, camera_meas = simulator.generate_measurement_joints_camera()
                measurements_actual = np.hstack([camera_meas, np.zeros((simulator.n, 3))])
                measurements_commanded = np.zeros((simulator.n, 6))
            else:
                # Generate standard measurements
                measurements_actual = []
                measurements_commanded = []
                for i in range(simulator.n):
                    actual, commanded, _, _ = simulator.generate_measurement_joints(calibrate=(j>0))
                    measurements_actual.append(actual)
                    measurements_commanded.append(commanded)
                measurements_actual = np.array(measurements_actual)
                measurements_commanded = np.array(measurements_commanded)
            
            # Compute Jacobians
            for i in range(simulator.n):
                simulator.compute_jacobians(simulator.joint_positions_actual[i])
            
            # Process results
            results_iter = simulator.process_iteration_results(
                measurements_actual, measurements_commanded,
                simulator.numJacobianTrans, simulator.numJacobianRot
            )
            
            final_errors.append(results_iter[0][0])  # Position error
        
        results[mode_name] = {
            'final_position_error': final_errors[-1],
            'convergence_rate': (final_errors[0] - final_errors[-1]) / final_errors[0] if final_errors[0] > 0 else 0,
            'parameter_count': simulator.numParameters
        }
        
        print(f"  Final position error: {results[mode_name]['final_position_error']:.6f} m")
        print(f"  Convergence rate: {results[mode_name]['convergence_rate']:.2%}")
        print(f"  Parameters estimated: {results[mode_name]['parameter_count']}")
    
    print(f"\n=== Comparison Summary ===")
    for mode in results:
        print(f"{mode} Mode:")
        print(f"  Final error: {results[mode]['final_position_error']:.6f} m")
        print(f"  Convergence: {results[mode]['convergence_rate']:.2%}")
        print(f"  Parameters: {results[mode]['parameter_count']}")

if __name__ == "__main__":
    print("Starting Camera-Based Calibration Test...")
    print("="*50)
    
    # Run main camera calibration test
    test_camera_calibration()
    
    # Optional: Run comparison test
    # compare_standard_vs_camera_mode()
    
    print("\nCamera calibration test completed!")