#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point
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



def main(args=None):
    rclpy.init(args=args)
    frame = "end_effector_link"
    use_joint_positions = 0
    robot = AR4_ROBOT(use_joint_positions)
    robot.resetErrors()
    marker_publisher = SurfacePublisher()

    simulator = CalibrationConvergenceSimulator()
    
    # Process each iteration separately
    for j in range(simulator.numIters):
        print(f"\n--- Starting Iteration {j} ---")
        simulator.set_current_iteration(j)
        
        # Pre-allocate numpy arrays for measurements
        measurements_actual = np.zeros((simulator.n, simulator.m))
        measurements_commanded = np.zeros((simulator.n, simulator.m))
        valid_measurements = 0

        # Generate individual measurements
        for i in range(simulator.n):
            counter = 0
            robot.moveit2.motion_suceeded = False
            while counter < 10:
                relativeToHomeAtGround, relativeToHomePos, globalHomePos = get_new_end_effector_position(robot)
            
                targetPos = np.array([globalHomePos[0], globalHomePos[1], 0])
                globalEndEffectorPos = relativeToHomePos+globalHomePos
                vectorToTarget = globalEndEffectorPos-targetPos
                roll, pitch, yaw = calculate_camera_pose(vectorToTarget[0], vectorToTarget[1], vectorToTarget[2])

                targetPosWeirdFrame, targetOrientWeirdFrame = robot.fromMyPreferredFrame(targetPos, np.array([0,0,0]), reference_frame="base_link")
                endEffectorPosWeirdFrame, endEffectorOrientWeirdFrame = robot.fromMyPreferredFrame(globalEndEffectorPos, np.array([roll,pitch,yaw]), reference_frame="base_link")
                
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
            actual = np.concatenate((endEffectorPosWeirdFrame, endEffectorOrientWeirdFrame))
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