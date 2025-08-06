#!/usr/bin/env python3
"""
Test script for camera-based calibration simulation

This script demonstrates how to use the calibration simulator in camera mode,
where measurements come from a camera mounted on the end effector measuring
displacement vectors to known targets in the environment.
"""

import rclpy
from ar4_robot_py import AR4Robot
from calibrationConvergenceSimulation import CalibrationSimulator
import numpy as np

def test_camera_calibration():
    """Test the calibration simulation using camera-based displacement measurements"""
    
    # Initialize ROS2
    
    try:
        # Create simulator with camera mode enabled
        print("Creating camera-based calibration simulator...")
        simulator = CalibrationSimulator(
            n=8,                    # Number of measurements per iteration
            numIters=5,             # Number of calibration iterations
            dQMagnitude=0.05,       # Joint angle error magnitude
            dLMagnitude=0.01,       # Link length error magnitude
            dXMagnitude=0.02,       # Base frame error magnitude
            camera_mode=True,       # Enable camera mode
            camera_transform=[0.05, 0.0, 0.02, 0.0, 0.0, 0.0]  # Camera mount: 5cm forward, 2cm up
        )
        
        # Create robot interface
        robot = AR4Robot()
        robot.disable_logging()
        
        print(f"Camera mode enabled: {simulator.camera_mode}")
        print(f"Camera transformation: {simulator.camera_transform}")
        print(f"Total parameters to estimate: {simulator.numParameters}")
        print(f"Parameter breakdown:")
        print(f"  - Joint errors (dQ): 6 parameters")
        print(f"  - Link length errors (dL): 6 parameters") 
        print(f"  - Base frame errors (dX): 6 parameters")
        print(f"Note: Camera transform is fixed/known - no additional parameters")
        print()
        
        # Print actual error values that we're trying to estimate
        print("Actual systematic errors to be estimated:")
        print(f"  Joint errors (dQ): {simulator.dQ}")
        print(f"  Link errors (dL): {simulator.dL}")
        print(f"  Base errors (dX): {simulator.dX}")
        print()
        
        # Define known target positions in world coordinates
        # These represent calibration targets placed in the workspace
        target_positions_world = np.array([
            [0.3, 0.2, 0.5],      # Target 1
            [0.2, -0.1, 0.6],     # Target 2
            [-0.1, 0.3, 0.4],     # Target 3
            [0.4, 0.0, 0.7],      # Target 4
            [0.0, 0.4, 0.5],      # Target 5
            [-0.2, -0.2, 0.6],    # Target 6
            [0.3, -0.3, 0.4],     # Target 7
            [-0.3, 0.1, 0.7],     # Target 8
        ])
        
        print("Target positions in world frame:")
        for i, target in enumerate(target_positions_world):
            print(f"  Target {i+1}: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
        print()
        
        # Process each calibration iteration
        for j in range(simulator.numIters):
            print(f"\n=== Camera Calibration Iteration {j+1}/{simulator.numIters} ===")
            simulator.set_current_iteration(j)
            
            # Generate camera-based measurements
            print("Generating camera displacement measurements...")
            joint_positions, target_displacements = simulator.generate_measurement_joints_camera(
                target_positions_world=target_positions_world
            )
            
            print(f"Generated {len(target_displacements)} camera displacement measurements")
            print(f"Sample displacement vectors (camera frame):")
            for i in range(min(3, len(target_displacements))):
                disp = target_displacements[i]
                print(f"  Measurement {i+1}: [{disp[0]:.4f}, {disp[1]:.4f}, {disp[2]:.4f}] m")
            
            # For each measurement, compute the Jacobian
            print("Computing Jacobians for camera measurements...")
            for i in range(simulator.n):
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_actual[i]
                )
                print(f"  Jacobian computed for measurement {i+1}")
            
            # In camera mode, measurements are displacement vectors
            # The differences are already computed in compute_differences method
            measurements_actual = np.zeros((simulator.n, 6))  # Dummy for interface
            measurements_commanded = np.zeros((simulator.n, 6))  # Dummy for interface
            
            # Process the iteration results
            print("Processing iteration results...")
            results = simulator.process_iteration_results(
                measurements_actual, 
                measurements_commanded,
                simulator.numJacobianTrans,
                simulator.numJacobianRot
            )
            
            avgTransAndRotError, dLEst, dQAct, dQEst, dXEst = results
            
            print(f"Results for iteration {j+1}:")
            print(f"  Average displacement error: {avgTransAndRotError[0]:.6f} m")
            
            # Show convergence progress
            joint_error_progress = np.linalg.norm(dQEst - simulator.dQ)
            base_error_progress = np.linalg.norm(dXEst - simulator.dX)
            link_error_progress = np.linalg.norm(dLEst - simulator.dL)
            
            print(f"  Joint parameter estimation error: {joint_error_progress:.6f}")
            print(f"  Base parameter estimation error: {base_error_progress:.6f}")
            print(f"  Link parameter estimation error: {link_error_progress:.6f}")
        
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
        
        print(f"Link errors (dL):")
        print(f"  True:      {simulator.dL}")
        print(f"  Estimated: {np.sum(simulator.dLMat, axis=0)}")
        print(f"  Error:     {np.sum(simulator.dLMat, axis=0) - simulator.dL}")
        
        # Save results to CSV
        simulator.save_to_csv('camera_calibration_results.csv')
        print(f"\nResults saved to 'camera_calibration_results.csv'")
        
        # Calculate final accuracy metrics
        final_joint_accuracy = np.linalg.norm(np.sum(simulator.dQMat, axis=0) - simulator.dQ)
        final_base_accuracy = np.linalg.norm(np.sum(simulator.dXMat, axis=0) - simulator.dX)
        final_link_accuracy = np.linalg.norm(np.sum(simulator.dLMat, axis=0) - simulator.dL)
        
        print(f"\nFinal Parameter Estimation Accuracy:")
        print(f"  Joint parameters: {final_joint_accuracy:.6f} rad RMS")
        print(f"  Base parameters: {final_base_accuracy:.6f} m/rad RMS")
        print(f"  Link parameters: {final_link_accuracy:.6f} m RMS")
        
        if final_joint_accuracy < 0.01 and final_base_accuracy < 0.005:
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

def test_camera_vs_standard_mode():
    """Quick test comparing camera mode vs standard mode"""
    
    print("\n=== Camera vs Standard Mode Test ===")
    
    # Test parameters
    test_params = {
        'n': 6,
        'numIters': 3,
        'dQMagnitude': 0.05,
        'dLMagnitude': 0.01,
        'dXMagnitude': 0.02
    }
    
    for mode_name, camera_mode, camera_transform in [
        ("Standard", False, None), 
        ("Camera", True, [0.05, 0.0, 0.02, 0.0, 0.0, 0.0])
    ]:
        print(f"\nTesting {mode_name} Mode...")
        
        # Create simulator
        if camera_mode:
            simulator = CalibrationSimulator(
                camera_mode=True, 
                camera_transform=camera_transform, 
                **test_params
            )
        else:
            simulator = CalibrationSimulator(camera_mode=False, **test_params)
        
        print(f"  Parameter count: {simulator.numParameters}")
        print(f"  Jacobian shape: {simulator.numJacobianTrans.shape[1] if simulator.numJacobianTrans.size > 0 else 'Not computed'}")
        
        if camera_mode:
            # Test camera measurement generation
            joints, displacements = simulator.generate_measurement_joints_camera()
            print(f"  Generated {len(displacements)} displacement measurements")
            print(f"  Sample displacement: {displacements[0]}")
        else:
            print(f"  Standard mode uses pose measurements")

if __name__ == "__main__":
    print("Starting Camera-Based Calibration Test...")
    print("="*50)
    
    # Run main camera calibration test
    test_camera_calibration()
    
    # Optional: Run comparison test (uncomment to enable)
    # test_camera_vs_standard_mode()
    
    print("\nCamera calibration test completed!")
