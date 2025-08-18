#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point, Pose
from surface_publisher import SurfacePublisher
from ar4_robot_py import AR4Robot
from calibrationConvergenceSimulation import CalibrationConvergenceSimulator
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
import csv
import os
import time
import cProfile
import pstats
import io

def get_new_end_effector_position(robot):
    random_numbers = np.random.rand(3) 
    random_pos = (random_numbers-0.5)*0.05
    xOffset = 0
    yOffset = 0
    zOffset = robot.pos_offsets["z"]
    x = xOffset + random_pos[0]
    y = yOffset + random_pos[1]
    z = zOffset + random_pos[2]
    relativeToHomeAtGround = np.array([x,y,z])

    globalHomePos = np.array([robot.pos_offsets["x"],robot.pos_offsets["y"],robot.pos_offsets["z"]])
    return relativeToHomeAtGround, random_pos, globalHomePos
def calculate_camera_pose(x, y, z, roll_offset=0.0, pitch_offset=0.0, yaw_offset=0.0):
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
    roll = 0.0
    
    # Apply offsets
    roll += roll_offset
    pitch += pitch_offset
    yaw += yaw_offset
    
    return roll, pitch, yaw


def print_current_position(robot,frame):
    current_pose = robot.get_current_pose(frame)
    position, orientation = current_pose
    robot.get_logger().info(f"Current Pose: Position={position}, Orientation={orientation}")

def camera_vector_from_pose_and_measurement(roll, pitch, yaw, distance):
    """
    Calculate a vector to target given orientation angles and distance.
    
    Parameters:
    - roll: Roll angle in radians
    - pitch: Pitch angle in radians
    - yaw: Yaw angle in radians
    - distance: Distance to the target
    
    Returns:
    - numpy array [x, y, z]: Vector pointing to the target
    """
    # First, understand the relationship between the angles and the direction vector
    # If we have pitch = -arctan2(x, z) - π/2, then x/z = -tan(pitch + π/2)
    # Similarly for roll = -arctan2(y, r), where r = sqrt(x^2 + z^2), then y/r = -tan(roll)
    
    # Calculate the vector components based on the spherical coordinates
    x_direction = -np.sin(pitch + np.pi/2) * np.cos(roll)
    y_direction = -np.sin(roll) * np.cos(pitch + np.pi/2)
    z_direction = np.cos(pitch + np.pi/2) * np.cos(roll)
    
    # Normalize the direction vector
    direction = np.array([x_direction, y_direction, z_direction])
    direction = direction / np.linalg.norm(direction)
    
    # Scale by the distance
    vector = direction * distance
    
    return vector

def main(args=None):
    frame = "end_effector_link"

    robot = AR4Robot()
    robot.disable_logging()
    marker_publisher = SurfacePublisher()
    # Create simulator with camera mode for visual demonstration
    simulator = CalibrationConvergenceSimulator(n=8, numIters=5, 
                                               dQMagnitude=0.1, dLMagnitude=0.0, 
                                               dXMagnitude=0.0, camera_mode=True)
    
    # Process each iteration separately
    for j in range(simulator.numIters):
        print(f"\n--- Starting Iteration {j} ---")
        simulator.set_current_iteration(j)

        # Generate individual measurements
        for i in range(simulator.n):
            counter = 0
            motionSucceeded = False
            while counter < 10:
                # Generate random end-effector position pointing toward target
                relativeToHomeAtGround, relativeToHomePos, globalHomePos = get_new_end_effector_position(robot)
                
                # Define target position (same as simulator's target)
                if simulator.camera_mode:
                    targetPosActual = np.array([0.3,0,0])
                    targetOrientActual = np.array([0,0,0])
                else:
                    targetPosActual = np.array([0,0,0])
                    targetOrientActual = np.array([0,0,0])

                globalEndEffectorPos = relativeToHomePos + globalHomePos
                
                # Calculate camera orientation to point at target
                # Vector should point FROM camera TO target
                vectorToTarget = targetPosActual - globalEndEffectorPos
                roll, pitch, yaw = calculate_camera_pose(vectorToTarget[0], 
                                                         vectorToTarget[1], 
                                                         vectorToTarget[2],
                                                         )
                
                # Generate measurement using the simulator's proper interface
                #pos_desired, orient_desired = robot.from_preferred_frame(position=globalEndEffectorPos, euler_angles=np.array([roll,pitch,yaw]),old_reference_frame="base_link", new_reference_frame="base_link")
                pose_desired = np.concatenate((globalEndEffectorPos, np.array([roll,pitch,yaw])))
                pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = simulator.generate_measurement_pose(
                    robot=robot, pose=pose_desired, calibrate=True, frame="base_link"
                )


                # Transform poses to robot's coordinate frame
                targetPosWeirdFrame, targetOrientWeirdFrame = robot.from_preferred_frame(
                    targetPosActual, targetOrientActual, 
                    old_reference_frame="base_link", new_reference_frame="base_link")
                '''endEffectorPosWeirdFrame, endEffectorOrientWeirdFrame = robot.from_preferred_frame(
                    globalEndEffectorPos, np.array([roll,pitch,yaw]), 
                    old_reference_frame="base_link", new_reference_frame="base_link")'''
                
                
                

                # Move robot to desired pose
                position = np.array([relativeToHomePos[0], relativeToHomePos[1], relativeToHomePos[2]])
                euler_angles = [roll, pitch, yaw]
                #euler_angles = np.array([0, 0, 0])
                #motionSucceeded = robot.move_to_pose_preferred_frame(pose_actual[:3], pose_actual[3:], frame)
                motionSucceeded = robot.move_to_joint_positions(joint_positions_actual)
                if motionSucceeded:
                    # Visualize target and camera-to-target vector
                    marker_publisher.publishPlane(np.array([0.146]), targetPosWeirdFrame)
                    marker_publisher.publish_arrow_between_points(
                    start=np.array([pose_actual[0], pose_actual[1], pose_actual[2]]),
                    end=np.array([targetPosWeirdFrame[0], targetPosWeirdFrame[1], targetPosWeirdFrame[2]]),
                    thickness=0.01,
                    id=1,
                    color=np.array([0.0, 1.0, 0.0])
                )
                    break
                counter += 1

            
            
            # Compute Jacobians properly based on mode
            if simulator.camera_mode:
                # Use the actual camera-to-target measurement
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample], 
                    camera_to_target=simulator.camera_to_target_actual)
            else:
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample])
                
            simulator.current_sample += 1  
            print(f"Measurement {i}: Generated successfully")
            time.sleep(0.01)

        # Process all measurements for this iteration using the correct target arrays
        results = simulator.process_iteration_results(
                simulator.targetPoseMeasured, 
                simulator.targetPoseExpected,
                simulator.numJacobianTrans,
                simulator.numJacobianRot)
        
        '''# Visualize convergence progress
        avgTransAndRotError, dLEst, dQAct, dQEst, dXEst = results
        print(f"Iteration {j} completed:")
        print(f"  Translation Error: {avgTransAndRotError[0]:.6f}")
        print(f"  Rotation Error: {avgTransAndRotError[1]:.6f}")
        print(f"  Joint Error Estimate: {dQEst}")
        print(f"  Base Offset Estimate: {dXEst}")'''
    
    # Save results to CSV
    #simulator.save_to_csv(filename='visual_calibration_data.csv')
    
    print('Visual calibration simulation completed!')
    rclpy.shutdown()

def profile_main():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your existing main() function call
    main()
    
    pr.disable()
    
    # Create a stats object and sort by cumulative time
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats(30)  # Show top 30 time-consuming functions
    #print(s.getvalue())

if __name__ == "__main__":
    profile_main()