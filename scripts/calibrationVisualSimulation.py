#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point, Pose
from surface_publisher import SurfacePublisher
from ar4_robot import AR4_ROBOT
from calibrationConvergenceSimulation import CalibrationConvergenceSimulator
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
import csv
import os
import time

def get_new_end_effector_position(robot):
    random_numbers = np.random.rand(3) 
    random_pos = (random_numbers-0.5)*0.1
    xOffset = 0
    yOffset = 0
    zOffset = robot.pos_offsets["z"]
    x = xOffset + random_pos[0]
    y = yOffset + random_pos[1]
    z = zOffset + random_pos[2]
    relativeToHomeAtGround = np.array([x,y,z])

    globalHomePos = np.array([robot.pos_offsets["x"],robot.pos_offsets["y"],robot.pos_offsets["z"]])
    return relativeToHomeAtGround, random_pos, globalHomePos
def calculate_camera_pose(x, y, z):
    pitch = -np.arctan2(x, z)-np.pi/2
    r = (x**2 + z**2)**0.5
    roll = -np.arctan2(y, r)
    yaw = 0
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
    rclpy.init(args=args)
    frame = "end_effector_link"
    use_joint_positions = 0
    robot = AR4_ROBOT(use_joint_positions)
    robot.resetErrors()
    marker_publisher = SurfacePublisher()

    simulator = CalibrationConvergenceSimulator()
    
    '''target_pose = Pose()
    target_pose.position.x = 0.0
    target_pose.position.y = 0.3
    target_pose.position.z = 0.3
    target_pose.orientation.x = 0.0
    target_pose.orientation.y = 0.0
    target_pose.orientation.z = 0.0
    target_pose.orientation.w = 1.0

    # Compute inverse kinematics
    joint_state = robot.moveit2.compute_ik(
        position=target_pose.position,
        quat_xyzw=target_pose.orientation,
    )'''
    
    #pos,ori = robot.get_current_pose()
    #joint_goals = robot.get_ik(pos,ori)
    
    # Process each iteration separately
    for j in range(simulator.numIters):
        print(f"\n--- Starting Iteration {j} ---")
        simulator.set_current_iteration(j)


        # Generate individual measurements
        for i in range(simulator.n):
            counter = 0
            robot.moveit2.motion_suceeded = False
            while counter < 10:
                relativeToHomeAtGround, relativeToHomePos, globalHomePos = get_new_end_effector_position(robot)
            
                
                targetPosBelieved = np.array([globalHomePos[0], globalHomePos[1], 0])
                targetPosActual = targetPosBelieved + np.array([0.05,0.05,0.05])
                globalEndEffectorPos = relativeToHomePos+globalHomePos
                vectorToTarget = globalEndEffectorPos-targetPosActual
                roll, pitch, yaw = calculate_camera_pose(vectorToTarget[0], vectorToTarget[1], vectorToTarget[2])
                
                distanceToTarget = np.linalg.norm(vectorToTarget)
                calculatedVector = camera_vector_from_pose_and_measurement(roll, pitch, yaw, distanceToTarget)
                measuredGlobalEndEffectorPos = targetPosBelieved + calculatedVector

                targetPosWeirdFrame, targetOrientWeirdFrame = robot.fromMyPreferredFrame(targetPosActual, np.array([0,0,0]), reference_frame="base_link")
                endEffectorPosWeirdFrame, endEffectorOrientWeirdFrame = robot.fromMyPreferredFrame(globalEndEffectorPos, np.array([roll,pitch,yaw]), reference_frame="base_link")
                measuredEndEffectorPosWeirdFrame, measuredEndEffectorOrientWeirdFrame = robot.fromMyPreferredFrame(measuredGlobalEndEffectorPos, np.array([roll,pitch,yaw]), reference_frame="base_link")

                marker_publisher.publishPlane(np.array([0.146]),targetPosWeirdFrame)

                marker_publisher.publish_arrow_between_points(
                    start=np.array([endEffectorPosWeirdFrame[0], endEffectorPosWeirdFrame[1], endEffectorPosWeirdFrame[2]]),  # Start at origin
                    end=np.array([targetPosWeirdFrame[0], targetPosWeirdFrame[1], targetPosWeirdFrame[2]]),   # End at point (1,1,1)
                    thickness=0.01,                   # 1cm thick shaft
                    id=20,                            # Unique ID
                    color=np.array([0.0, 1.0, 0.0])   # Red color
                )

                #print_current_position(robot,frame)
                
                position = [relativeToHomePos[0], relativeToHomePos[1], relativeToHomePos[2]]
                euler_angles = [roll,pitch,yaw]
                robot.move_to_pose(position, euler_angles,frame)
                if robot.moveit2.motion_suceeded:
                    break
                counter += 1


            #actual, commanded = simulator.generate_measurement(i)
            actual = np.concatenate((measuredEndEffectorPosWeirdFrame, measuredEndEffectorOrientWeirdFrame))
            commanded = np.concatenate((endEffectorPosWeirdFrame, endEffectorOrientWeirdFrame))
            if actual is not None and commanded is not None:
                measurements_actual[valid_measurements] = actual
                measurements_commanded[valid_measurements] = commanded
                valid_measurements += 1
                print(f"Measurement {i}: Generated")
            time.sleep(0.01)

        # Trim arrays to only include valid measurements
        if valid_measurements < simulator.n:
            measurements_actual = measurements_actual[:valid_measurements]
            measurements_commanded = measurements_commanded[:valid_measurements]
        
        # Process all measurements for this iteration
        results = simulator.process_iteration_results(measurements_actual, measurements_commanded)
    
    # Save results to CSV
    #simulator.save_to_csv()
    
    
        
       



if __name__ == "__main__":
    main()