#!/usr/bin/env python3

import rclpy
from ar4_robot_py import AR4Robot
import numpy as np

def test_ik_simple():
    """Simple test of the get_ik function"""
    try:
        # Create robot interface
        robot = AR4Robot()
        
        print("=== Testing IK function ===")
        
        # Test: IK in base_link frame with simple pose
        print("\n--- Test: IK in base_link frame ---")
        position = [0.3, 0.0, 0.3]  # Simple reachable position
        orientation = [0.0, 0.0, 0.0]  # No rotation
        
        print(f"Testing position: {position}")
        print(f"Testing orientation: {orientation}")
        
        ik_result = robot.get_ik(
            position=position,
            euler_angles=orientation,
            frame_id="base_link"
        )
        
        if ik_result:
            print("SUCCESS: IK solution found!")
            for joint, value in ik_result.items():
                print(f"  {joint}: {np.degrees(value):.1f}°")
        else:
            print("FAILED: No IK solution found")
        
        # Test: IK in end_effector_link frame
        print("\n--- Test: IK in end_effector_link frame ---")
        position_ee = [0.01, 0.0, 0.0]  # Small offset from current end effector
        orientation_ee = [0.0, 0.0, 0.0]
        
        print(f"Testing position: {position_ee}")
        print(f"Testing orientation: {orientation_ee}")
        
        ik_result_ee = robot.get_ik(
            position=position_ee,
            euler_angles=orientation_ee,
            frame_id="end_effector_link"
        )
        
        if ik_result_ee:
            print("SUCCESS: IK solution found for end_effector_link frame!")
            for joint, value in ik_result_ee.items():
                print(f"  {joint}: {np.degrees(value):.1f}°")
        else:
            print("FAILED: No IK solution for end_effector_link frame")
        
        print("\n=== Test completed ===")
        
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.shutdown()

if __name__ == '__main__':
    test_ik_simple()
