#!/usr/bin/env python3

import rclpy
from ar4_robot_py import AR4Robot
import numpy as np

def test_ik_frames():
    """Test the get_ik function with different frames"""
    try:
        # Create robot interface
        robot = AR4Robot()
        
        print("=== Testing IK with different frames ===")
        
        # Test 1: IK in base_link frame (should work directly)
        print("\n--- Test 1: IK in base_link frame ---")
        position_base = [0.3, 0.0, 0.4]
        orientation_base = [0.0, 0.0, 0.0]
        
        ik_result_base = robot.get_ik(
            position=position_base,
            euler_angles=orientation_base,
            frame_id="base_link"
        )
        
        if ik_result_base:
            print("SUCCESS: IK solution found for base_link frame")
            for joint, value in ik_result_base.items():
                print(f"  {joint}: {np.degrees(value):.1f}°")
        else:
            print("FAILED: No IK solution for base_link frame")
        
        # Test 2: IK in end_effector_link frame (should convert then solve)
        print("\n--- Test 2: IK in end_effector_link frame ---")
        position_ee = [0.0, 0.0, 0.0]  # Small offset from current end effector
        orientation_ee = [0.0, 0.0, 0.0]
        
        ik_result_ee = robot.get_ik(
            position=position_ee,
            euler_angles=orientation_ee,
            frame_id="end_effector_link"
        )
        
        if ik_result_ee:
            print("SUCCESS: IK solution found for end_effector_link frame")
            for joint, value in ik_result_ee.items():
                print(f"  {joint}: {np.degrees(value):.1f}°")
        else:
            print("FAILED: No IK solution for end_effector_link frame")
        
        
        
        print("\n=== IK frame tests completed ===")
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        robot.shutdown()

if __name__ == '__main__':
    test_ik_frames()
